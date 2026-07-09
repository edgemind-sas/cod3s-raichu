"""Python-callback twin of bench.cpp — same models, same PyCATSHOO
1.4.1.0 engine, same simulation configuration; only the modelling
language differs (conditions, sensitive methods and ODE right-hand
sides are Python callbacks, the normal PyCATSHOO modelling style).

With the same seed, the C++ and Python variants must produce the same
estimates — that is the consistency check; the wall-clock difference
is then exactly the Python-boundary cost.

Run with a Python matching the Pycatshoo module (3.11 for 1.4.1.0):
  PYTHONPATH=$PYCATSHOO_DIR/Core/lib python3.11 bench_py.py <model> <nb_runs> <t_max> <seed>
"""

import json
import sys
import time

import Pycatshoo as Pyc

# Tutorial constants — single numeric source is
# python/tests/validation/fixtures/gen_tutorial_fixtures.py.
LAMBDA, MU, NOMINAL = 0.01, 0.1, 5.0
T_MIN, T_MAX_THR = 15.0, 20.0
LEAKAGE, T_OUT, T_INIT = 0.1, 13.0, 17.0


# --8<-- [start:pureexp]
class PureExp(Pyc.CComponent):
    def __init__(self, name):
        Pyc.CComponent.__init__(self, name)
        lam = self.addVariable("lambda", Pyc.TVarType.t_double, LAMBDA)
        mu = self.addVariable("mu", Pyc.TVarType.t_double, MU)
        self.addAutomaton("aut")
        state_ok = self.addState("aut", "OK", 1)
        state_ko = self.addState("aut", "KO", 0)
        self.setInitState("OK")
        trans = state_ok.addTransition("fail")
        trans.setDistLaw(Pyc.IDistLaw.newLaw(self, Pyc.TLawType.expo, lam))
        trans.addTarget(state_ko, Pyc.TTransType.fault)
        trans = state_ko.addTransition("repair")
        trans.setDistLaw(Pyc.IDistLaw.newLaw(self, Pyc.TLawType.expo, mu))
        trans.addTarget(state_ok, Pyc.TTransType.rep)
# --8<-- [end:pureexp]


class Heater(Pyc.CComponent):
    def __init__(self, name, init_on, with_room):
        Pyc.CComponent.__init__(self, name)
        self.with_room = with_room
        self.po_nominalPower = self.addVariable(
            "nominalPower", Pyc.TVarType.t_double, NOMINAL
        )
        self.po_power = self.addVariable(
            "power", Pyc.TVarType.t_double, NOMINAL if init_on else 0.0
        )
        self.po_lambda = self.addVariable("lambda", Pyc.TVarType.t_double, LAMBDA)
        self.po_mu = self.addVariable("mu", Pyc.TVarType.t_double, MU)
        self.pi_startingRequest = self.addReference("startingRequest")
        if with_room:
            self.po_minTemperature = self.addVariable(
                "minTemperature", Pyc.TVarType.t_double, T_MIN
            )
            self.po_maxTemperature = self.addVariable(
                "maxTemperature", Pyc.TVarType.t_double, T_MAX_THR
            )
            self.pi_roomTemperature = self.addReference("roomTemperature")

        self.addAutomaton("DysfunctionalAutomaton")
        self.stateOK = self.addState("DysfunctionalAutomaton", "OK", 1)
        self.stateKO = self.addState("DysfunctionalAutomaton", "KO", 0)
        self.setInitState("OK")
        trans = self.stateOK.addTransition("OK_to_KO")
        trans.setDistLaw(Pyc.IDistLaw.newLaw(self, Pyc.TLawType.expo, self.po_lambda))
        trans.addTarget(self.stateKO, Pyc.TTransType.fault)
        trans = self.stateKO.addTransition("KO_to_OK")
        trans.setDistLaw(Pyc.IDistLaw.newLaw(self, Pyc.TLawType.expo, self.po_mu))
        trans.addTarget(self.stateOK, Pyc.TTransType.rep)

        functional = self.addAutomaton("FunctionalAutomaton")
        self.stateON = self.addState("FunctionalAutomaton", "ON", 1)
        self.stateOFF = self.addState("FunctionalAutomaton", "OFF", 0)
        self.setInitState("ON" if init_on else "OFF")
        functional.addCallback("updateSuppliedPower")
        self.transOff2On = self.stateOFF.addTransition("OFF_to_ON")
        self.transOff2On.setCondition(self.off2OnCondition)
        self.transOff2On.addTarget(self.stateON, Pyc.TTransType.trans)
        self.transOn2Off = self.stateON.addTransition("ON_to_OFF")
        self.transOn2Off.setCondition(self.on2OffCondition)
        self.transOn2Off.addTarget(self.stateOFF, Pyc.TTransType.trans)

        box = self.addMessageBox("MB-toMaster")
        box.addImport(self.pi_startingRequest, "request")
        box = self.addMessageBox("MB-toSlave")
        box.addExport(self.stateKO, "request")
        if with_room:
            box = self.addMessageBox("MB-toRoom")
            box.addExport(self.po_power, "power")
            box.addImport(self.pi_roomTemperature, "temperature")

        self.addStartMethod("UpdateSuppliedPower", self.updateSuppliedPower)

    def off2OnCondition(self):
        if not self.stateOK.isActive():
            return False
        if not self.pi_startingRequest.orValue(True):
            return False
        if self.with_room:
            return self.pi_roomTemperature.dValue(0) < self.po_minTemperature.dValue()
        return True

    def on2OffCondition(self):
        if not self.stateOK.isActive():
            return True
        if not self.pi_startingRequest.orValue(True):
            return True
        if self.with_room:
            return self.pi_roomTemperature.dValue(0) > self.po_maxTemperature.dValue()
        return False

    def updateSuppliedPower(self):
        if self.stateON.isActive():
            self.po_power.setDValue(self.po_nominalPower.dValue())
        else:
            self.po_power.setDValue(0.0)


class Room(Pyc.CComponent):
    def __init__(self, name):
        Pyc.CComponent.__init__(self, name)
        self.po_leakageRate = self.addVariable(
            "leakageRate", Pyc.TVarType.t_double, LEAKAGE
        )
        self.po_temperature = self.addVariable(
            "temperature", Pyc.TVarType.t_double, T_INIT
        )
        self.po_outsideTemperature = self.addVariable(
            "outsideTemperature", Pyc.TVarType.t_double, T_OUT
        )
        self.pi_power = self.addReference("power")
        box = self.addMessageBox("MB-toHeaters")
        box.addImport(self.pi_power, "power")
        box.addExport(self.po_temperature, "temperature")

    def pdmpMethod(self):
        self.po_temperature.setDvdtODE(
            self.pi_power.sumValue(0.0)
            - self.po_leakageRate.dValue()
            * (self.po_temperature.dValue() - self.po_outsideTemperature.dValue())
        )


class PureExpSystem(Pyc.CSystem):
    def __init__(self):
        Pyc.CSystem.__init__(self, "PureExp")
        self.component = PureExp("C")


class HeatersS1System(Pyc.CSystem):
    def __init__(self):
        Pyc.CSystem.__init__(self, "S1")
        self.master = Heater("aMasterHeater", init_on=True, with_room=False)


class HeatedRoomS3System(Pyc.CSystem):
    def __init__(self):
        Pyc.CSystem.__init__(self, "S3")
        pdmp = self.addPDMPManager("PDMP-Manager")
        pdmp.setDtCond(1e-6)
        self.master = Heater("aMasterHeater", init_on=True, with_room=True)
        self.slave = Heater("aSlaveHeater", init_on=False, with_room=True)
        self.connect("aMasterHeater", "MB-toSlave", "aSlaveHeater", "MB-toMaster")
        self.room = Room("Room")
        pdmp.addEquationMethod("pdmpMethod", self.room)
        pdmp.addODEVariable(self.room.po_temperature)
        for heater in (self.master, self.slave):
            pdmp.addWatchedTransition(heater.transOff2On)
            pdmp.addWatchedTransition(heater.transOn2Off)
        self.connect("aMasterHeater", "MB-toRoom", "Room", "MB-toHeaters")
        self.connect("aSlaveHeater", "MB-toRoom", "Room", "MB-toHeaters")


MODELS = {
    "pure_exp": (PureExpSystem, [("C_KO", "C.KO", "ST")]),
    "heaters_s1": (HeatersS1System, [("aHeater_power", "aMasterHeater.power", "VAR")]),
    "heated_room_s3": (
        HeatedRoomS3System,
        [
            ("aMasterHeater_power", "aMasterHeater.power", "VAR"),
            ("aSlaveHeater_power", "aSlaveHeater.power", "VAR"),
            ("Room_temperature", "Room.temperature", "VAR"),
        ],
    ),
}


def main():
    model, nb_runs, t_max, seed = (
        sys.argv[1],
        int(sys.argv[2]),
        float(sys.argv[3]),
        int(sys.argv[4]),
    )
    cls, specs = MODELS[model]
    system = cls()

    indicators = []
    for name, element, element_type in specs:
        indicator = system.addIndicator(name, element, element_type)
        indicator.setRestitutions(
            Pyc.TIndicatorType.mean_values | Pyc.TIndicatorType.std_dev
        )
        indicators.append(indicator)

    n_values = 11
    for i in range(n_values):
        system.addInstant(t_max * i / (n_values - 1))
    system.setTMax(t_max)
    system.setRNGSeed(seed)
    system.setNbSeqToSim(nb_runs)
    system.setRNG("yarn5")
    system.setRNGBlockSize(1000)

    started = time.perf_counter()
    system.simulate()
    wall = time.perf_counter() - started

    print(
        json.dumps(
            {
                "model": model,
                "nb_runs": nb_runs,
                "seed": seed,
                "wall_clock_s": wall,
                "instants": list(system.instants()),
                "estimates": {
                    name: {
                        "mean": [float(v) for v in ind.means()],
                        "std": [float(v) for v in ind.stdDevs()],
                    }
                    for (name, _, _), ind in zip(specs, indicators)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
