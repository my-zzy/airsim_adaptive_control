# airsim_adaptive_control
Quadrotor simulation in Airsim using adaptive control

## main problem
attitude control has too much overshoot which makes position control impossible

## core code
|file name|function|remark|
|:------:|:------:|:------:|
|analytical_model|physics model|shouldn't change, NED|
|run_in_analytical_model|main script for position control|test fail|
|run_att_in_analytical_model|script for attitude control|overshoot|
|controller_old|all controllers||
|controller_old|controllers with filters|written by ai|
|config|all parameters||
|traj|trajectory||

## debug code
|file name|function|remark|
|:------:|:------:|:------:|
|coordinate_analysis|||
|debug_control|||
|debug_dynamics|||
|filter|||
|PARAMETER_TESTING_GUIDE||.md|
|test_gravity_thrust|||

## unrelated
the rest