#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

extern void raichu_fmi_record_log(void *environment, const char *message);

void raichu_fmi2_logger(void *environment, const char *instance_name,
                       int status, const char *category, const char *format, ...) {
    (void)instance_name;
    (void)status;
    (void)category;
    char buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    raichu_fmi_record_log(environment, buffer);
}
