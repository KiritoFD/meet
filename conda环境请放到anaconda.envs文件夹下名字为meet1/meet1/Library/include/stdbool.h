#ifndef _STDBOOL_H
#define _STDBOOL_H

#ifndef __cplusplus
#define bool  int
#define true  1
#define false 0
#else
#define bool  bool
#define false false
#define true  true
#endif

#define __bool_true_false_are_defined  1

#endif

