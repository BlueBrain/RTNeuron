#include "config.h"
#include <stdlib.h>
#include "gsl/gsl_vector.h"
#include "gsl/view.h"

#define BASE_LONG_DOUBLE
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_LONG_DOUBLE

#define BASE_DOUBLE
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_DOUBLE

#define BASE_FLOAT
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_FLOAT

#define USE_QUALIFIER
#define QUALIFIER const

#define BASE_LONG_DOUBLE
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_LONG_DOUBLE

#define BASE_DOUBLE
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_DOUBLE

#define BASE_FLOAT
#include "gsl/templates_on.h"
#include "gsl/view_source.c"
#include "gsl/templates_off.h"
#undef  BASE_FLOAT
