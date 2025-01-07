\ Model Sub_normal_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - u_dual_sub[0] - u_dual_sub[1] - u_dual_sub[2] - u_dual_sub[3]
   - u_dual_sub[4] - u_dual_sub[5] + u_dual_sub[8]
Subject To
 Subproblem_constraints[0]: - u_dual_sub[0] - u_dual_sub[1] - u_dual_sub[4]
   <= -8
 Subproblem_constraints[1]: - u_dual_sub[2] - u_dual_sub[3] - u_dual_sub[5]
   <= -9
 Subproblem_constraints[2]: - u_dual_sub[0] - u_dual_sub[2] - u_dual_sub[6]
   <= -5
 Subproblem_constraints[3]: - u_dual_sub[1] - u_dual_sub[3] - u_dual_sub[7]
   <= -6
 Subproblem_constraints[4]: u_dual_sub[8] <= 0
Bounds
 u_dual_sub[8] free
End
