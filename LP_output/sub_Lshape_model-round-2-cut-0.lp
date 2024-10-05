\ Model Sub_Lshape
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  u_sub[0] + vr_sub[0] + vr_sub[1] + vr_sub[2] + vr_sub[3] + vr_sub[4]
   + vr_sub[5] + vr_sub[6] + vr_sub[7] + vr_sub[8]
Subject To
 Sub_Ls_problem_ieq_constraints_0: vr_sub[0] - y_sub[0] - y_sub[2] >= -1
 Sub_Ls_problem_ieq_constraints_1: vr_sub[1] - y_sub[0] - y_sub[3] >= -1
 Sub_Ls_problem_ieq_constraints_2: vr_sub[2] - y_sub[1] - y_sub[2] >= -1
 Sub_Ls_problem_ieq_constraints_3: vr_sub[3] - y_sub[1] - y_sub[3] >= -1
 Sub_Ls_problem_ieq_constraints_4: vr_sub[4] - y_sub[0] >= -1
 Sub_Ls_problem_ieq_constraints_5: vr_sub[5] - y_sub[1] >= -1
 Sub_Ls_problem_ieq_constraints_6: vr_sub[6] - y_sub[2] >= 0
 Sub_Ls_problem_ieq_constraints_7: vr_sub[7] - y_sub[3] >= 0
 Sub_Ls_problem_eq_constraints_8: u_sub[0] - vr_sub[8] + y_sub[4] = 1
Bounds
End
