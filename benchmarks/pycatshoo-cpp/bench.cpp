// PyCATSHOO C++-native benchmark models — fairness companion to
// python/tests/validation/perf_bench.py.
//
// The Python oracle drives PyCATSHOO through models whose conditions,
// sensitive methods, boundary checkers and ODE right-hand sides are
// Python callbacks: the C++ engine crosses the interpreter boundary at
// every evaluation. This program rebuilds the same three models fully
// in C++ against libPycatshoo.so, so the whole Monte-Carlo hot loop
// stays native — the strictest apples-to-apples baseline available for
// RAICHU comparisons.
//
// Faithfulness contract (mirrors oracles/run_oracle.py builders):
//   - identical structure, parameters, message boxes and conditions;
//   - identical simulation configuration: RNG yarn5, bloc size 1000,
//     seed via argv, 11 indicator instants over [0, t_max];
//   - indicators mean|std_dev, computation `simple`;
//   - wall-clock covers simulate() only (construction excluded), like
//     capture_mc() on the oracle side.
//
// Usage: pyc_bench <pure_exp|heaters_s1|heated_room_s3> <nb_runs> <t_max> <seed>
// Output: one JSON document on stdout.
//
// Built against the official PyCATSHOO 1.4.1.0 Linux distribution
// (Core/include/PyC + libPycatshoo.so), the only distribution on this
// machine whose headers match its binaries — the 1.3.8.0 install under
// Dev/pycatshoo ships 1.3.7.x-era headers whose class layouts drifted
// from the .so (subclassing crashes). The Python side of the
// same-version comparison (bench_py.py) runs on the same 1.4.1.0
// Pycatshoo module.

#include "System.h"
#include "Component.h"
#include "Interfaces.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// libPycatshoo.a bundles the FMU wrapper objects, which expect the
// embedding knowledge base to provide getKB(); the FMU entry points
// are never reached in this benchmark.
extern "C" void* getKB() { return nullptr; }

// Tutorial constants — single numeric source is
// python/tests/validation/fixtures/gen_tutorial_fixtures.py; keep in
// sync with LAMBDA, MU, NOMINAL, T_MIN, T_MAX_THR, LEAKAGE, T_OUT,
// T_INIT there.
namespace params {
// kLambda is a runtime knob: the tolerance-parity experiment uses a
// deterministic declination (lambda = 0, pure thermostat cycle with a
// closed-form trajectory) to measure each engine's achieved accuracy.
double kLambda = 0.01;
constexpr double kMu = 0.1;
constexpr double kNominal = 5.0;
constexpr double kTMin = 15.0;
constexpr double kTMaxThr = 20.0;
constexpr double kLeakage = 0.1;
constexpr double kTOut = 13.0;
constexpr double kTInit = 17.0;
// PDMP integration effort (negative = engine default). PyCATSHOO's
// defaults: RK4 with dt = 0.01, condition search dtCond = 0.001; the
// oracle and bench baseline set dtCond = 1e-6.
double kDt = -1.0;
double kDtCond = 1e-6;
}  // namespace params

// --- pure_exp: two-state exponential component, zero callbacks -------------

// --8<-- [start:pureexp]
class PureExp : public CComponent {
 public:
  IState* stateOK;
  IState* stateKO;

  PureExp(char const* name, CSystem& system) : CComponent(name, system) {
    IVariable* lambda =
        addVariable("lambda", TVarType::t_double, params::kLambda);
    IVariable* mu = addVariable("mu", TVarType::t_double, params::kMu);
    addAutomaton("aut");
    stateOK = addState("aut", "OK", 1);
    stateKO = addState("aut", "KO", 0);
    setInitState(stateOK);
    ITransition* trans = stateOK->addTransition("fail");
    trans->setDistLaw(TLawType::expo, *lambda);
    trans->addTarget(stateKO, TTransType::fault);
    trans = stateKO->addTransition("repair");
    trans->setDistLaw(TLawType::expo, *mu);
    trans->addTarget(stateOK, TTransType::rep);
  }
};
// --8<-- [end:pureexp]

// --- the tutorial heater (oracle _tutorial_heater_class, law="exp") --------

class Heater : public CComponent {
 public:
  IVariable* poNominalPower;
  IVariable* poPower;
  IVariable* poLambda;
  IVariable* poMu;
  IVariable* poMinTemperature = nullptr;
  IVariable* poMaxTemperature = nullptr;
  IReference* piStartingRequest;
  IReference* piRoomTemperature = nullptr;
  IState* stateOK;
  IState* stateKO;
  IState* stateON;
  IState* stateOFF;
  ITransition* transOff2On;
  ITransition* transOn2Off;
  bool const withRoom;

  Heater(char const* name, CSystem& system, bool initOn, bool room)
      : CComponent(name, system), withRoom(room) {
    poNominalPower =
        addVariable("nominalPower", TVarType::t_double, params::kNominal);
    poPower = addVariable("power", TVarType::t_double,
                          initOn ? params::kNominal : 0.0);
    poLambda = addVariable("lambda", TVarType::t_double, params::kLambda);
    poMu = addVariable("mu", TVarType::t_double, params::kMu);
    piStartingRequest = addReference("startingRequest");
    if (withRoom) {
      poMinTemperature =
          addVariable("minTemperature", TVarType::t_double, params::kTMin);
      poMaxTemperature =
          addVariable("maxTemperature", TVarType::t_double, params::kTMaxThr);
      piRoomTemperature = addReference("roomTemperature");
    }

    addAutomaton("DysfunctionalAutomaton");
    stateOK = addState("DysfunctionalAutomaton", "OK", 1);
    stateKO = addState("DysfunctionalAutomaton", "KO", 0);
    setInitState(stateOK);
    ITransition* trans = stateOK->addTransition("OK_to_KO");
    trans->setDistLaw(TLawType::expo, *poLambda);
    trans->addTarget(stateKO, TTransType::fault);
    trans = stateKO->addTransition("KO_to_OK");
    trans->setDistLaw(TLawType::expo, *poMu);
    trans->addTarget(stateOK, TTransType::rep);

    IAutomaton* functional = addAutomaton("FunctionalAutomaton");
    stateON = addState("FunctionalAutomaton", "ON", 1);
    stateOFF = addState("FunctionalAutomaton", "OFF", 0);
    setInitState(initOn ? stateON : stateOFF);
    functional->addCallback("updateSuppliedPower",
                            &Heater::updateSuppliedPower);
    transOff2On = stateOFF->addTransition("OFF_to_ON");
    transOff2On->setCondition(&Heater::off2OnCondition);
    transOff2On->addTarget(stateON, TTransType::trans);
    transOn2Off = stateON->addTransition("ON_to_OFF");
    transOn2Off->setCondition(&Heater::on2OffCondition);
    transOn2Off->addTarget(stateOFF, TTransType::trans);

    IMessageBox* box = addMessageBox("MB-toMaster");
    box->addImport(piStartingRequest, "request");
    box = addMessageBox("MB-toSlave");
    box->addExport(stateKO, "request");
    if (withRoom) {
      box = addMessageBox("MB-toRoom");
      box->addExport(poPower, "power");
      box->addImport(piRoomTemperature, "temperature");
    }

    addStartMethod("UpdateSuppliedPower", &Heater::updateSuppliedPower);
  }

  bool off2OnCondition() {
    if (!stateOK->isActive()) return false;
    if (!piStartingRequest->orValue(true)) return false;
    if (withRoom)
      return piRoomTemperature->dValue(0) < poMinTemperature->dValue();
    return true;
  }

  bool on2OffCondition() {
    if (!stateOK->isActive()) return true;
    if (!piStartingRequest->orValue(true)) return true;
    if (withRoom)
      return piRoomTemperature->dValue(0) > poMaxTemperature->dValue();
    return false;
  }

  void updateSuppliedPower() {
    poPower->setValue(stateON->isActive() ? poNominalPower->dValue() : 0.0);
  }
};

// --- the tutorial room (oracle Room inner class) ----------------------------

class Room : public CComponent {
 public:
  IVariable* poLeakageRate;
  IVariable* poTemperature;
  IVariable* poOutsideTemperature;
  IReference* piPower;

  Room(char const* name, CSystem& system) : CComponent(name, system) {
    poLeakageRate =
        addVariable("leakageRate", TVarType::t_double, params::kLeakage);
    poTemperature =
        addVariable("temperature", TVarType::t_double, params::kTInit);
    poOutsideTemperature =
        addVariable("outsideTemperature", TVarType::t_double, params::kTOut);
    piPower = addReference("power");
    IMessageBox* box = addMessageBox("MB-toHeaters");
    box->addImport(piPower, "power");
    box->addExport(poTemperature, "temperature");
  }

  void pdmpMethod() {
    poTemperature->setDvdtODE(
        piPower->sumValue(0.0) -
        poLeakageRate->dValue() *
            (poTemperature->dValue() - poOutsideTemperature->dValue()));
  }
};

// --- systems ----------------------------------------------------------------

class PureExpSystem : public CSystem {
 public:
  explicit PureExpSystem() : CSystem("PureExp") { new PureExp("C", *this); }
};

class HeatersS1System : public CSystem {
 public:
  explicit HeatersS1System() : CSystem("S1") {
    new Heater("aMasterHeater", *this, /*initOn=*/true, /*room=*/false);
  }
};

class HeatedRoomS3System : public CSystem {
 public:
  explicit HeatedRoomS3System() : CSystem("S3") {
    IPDMPManager* pdmp = addPDMPManager("PDMP-Manager");
    pdmp->setDtCond(params::kDtCond);
    if (params::kDt > 0.0) pdmp->setDt(params::kDt);
    Heater* master =
        new Heater("aMasterHeater", *this, /*initOn=*/true, /*room=*/true);
    Heater* slave =
        new Heater("aSlaveHeater", *this, /*initOn=*/false, /*room=*/true);
    connect("aMasterHeater", "MB-toSlave", "aSlaveHeater", "MB-toMaster");
    Room* room = new Room("Room", *this);
    pdmp->addEquationMethod("pdmpMethod", *room, &Room::pdmpMethod);
    pdmp->addODEVariable(*room->poTemperature);
    for (Heater* heater : {master, slave}) {
      pdmp->addWatchedTransition(*heater->transOff2On);
      pdmp->addWatchedTransition(*heater->transOn2Off);
    }
    connect("aMasterHeater", "MB-toRoom", "Room", "MB-toHeaters");
    connect("aSlaveHeater", "MB-toRoom", "Room", "MB-toHeaters");
  }
};

// --- driver -----------------------------------------------------------------

struct IndicatorSpec {
  char const* name;         // report key (matches the fixture indicator)
  char const* element;      // PyCATSHOO composite element name
  char const* elementType;  // VAR | ST
};

int main(int argc, char** argv) {
  if (argc < 5 || argc > 8) {
    std::fprintf(
        stderr,
        "usage: %s <pure_exp|heaters_s1|heated_room_s3> <nb_runs> <t_max> "
        "<seed> [dt] [dtCond] [lambda]\n"
        "  dt/dtCond: PDMP integration effort (dt <= 0 = engine default)\n"
        "  lambda: heater failure rate (0 = deterministic thermostat "
        "cycle)\n",
        argv[0]);
    return 2;
  }
  std::string const model = argv[1];
  long long const nbRuns = std::atoll(argv[2]);
  double const tMax = std::atof(argv[3]);
  unsigned const seed = static_cast<unsigned>(std::atoi(argv[4]));
  if (argc > 5) params::kDt = std::atof(argv[5]);
  if (argc > 6) params::kDtCond = std::atof(argv[6]);
  if (argc > 7) params::kLambda = std::atof(argv[7]);

  CSystem* system = nullptr;
  std::vector<IndicatorSpec> specs;
  if (model == "pure_exp") {
    system = new PureExpSystem();
    specs = {{"C_KO", "C.KO", "ST"}};
  } else if (model == "heaters_s1") {
    system = new HeatersS1System();
    specs = {{"aHeater_power", "aMasterHeater.power", "VAR"}};
  } else if (model == "heated_room_s3") {
    system = new HeatedRoomS3System();
    specs = {{"aMasterHeater_power", "aMasterHeater.power", "VAR"},
             {"aSlaveHeater_power", "aSlaveHeater.power", "VAR"},
             {"Room_temperature", "Room.temperature", "VAR"}};
  } else {
    std::fprintf(stderr, "unknown model: %s\n", model.c_str());
    return 2;
  }

  std::vector<IIndicator*> indicators;
  for (IndicatorSpec const& spec : specs) {
    IIndicator* indicator =
        system->addIndicator(spec.name, spec.element, spec.elementType);
    indicator->setRestitutions(TIndicatorType::mean_values |
                               TIndicatorType::std_dev);
    indicators.push_back(indicator);
  }

  // Mirror cod3s prepare_simu(): 11 instants over [0, t_max], yarn5
  // with bloc size 1000, fixed seed.
  int const kNValues = 11;
  for (int i = 0; i < kNValues; ++i)
    system->addInstant(tMax * i / (kNValues - 1));
  system->setTMax(tMax);
  system->setRNGSeed(seed);
  system->setNbSeqToSim(nbRuns);
  system->setRNG("yarn5");
  system->setRNGBlockSize(1000);

  auto const started = std::chrono::steady_clock::now();
  system->simulate();
  double const wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();

  std::vector<double> const instants = system->instants();
  std::printf("{\n  \"model\": \"%s\",\n  \"nb_runs\": %lld,\n", model.c_str(),
              nbRuns);
  std::printf("  \"seed\": %u,\n  \"wall_clock_s\": %.9f,\n", seed, wall);
  std::printf("  \"instants\": [");
  for (size_t i = 0; i < instants.size(); ++i)
    std::printf("%s%g", i ? ", " : "", instants[i]);
  std::printf("],\n  \"estimates\": {\n");
  for (size_t k = 0; k < indicators.size(); ++k) {
    std::vector<float> const means = indicators[k]->means();
    std::vector<float> const stds = indicators[k]->stdDevs();
    std::printf("    \"%s\": {\"mean\": [", specs[k].name);
    for (size_t i = 0; i < means.size(); ++i)
      std::printf("%s%.9g", i ? ", " : "", static_cast<double>(means[i]));
    std::printf("], \"std\": [");
    for (size_t i = 0; i < stds.size(); ++i)
      std::printf("%s%.9g", i ? ", " : "", static_cast<double>(stds[i]));
    std::printf("]}%s\n", k + 1 < indicators.size() ? "," : "");
  }
  std::printf("  }\n}\n");
  return 0;
}
