import os
import torch
import matplotlib.pyplot as plt
import pickle 
from libs import thermodynamics as tmd 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONTINUATION METHOD
@torch.no_grad()
def continuation(
	sol_0,
	model,
	ds = 0.3,
	continuation_steps = 10,
	max_corrector_steps = 20,
	plotdir = None,
	alpha = 0.5,
	**kwargs,
	):
	"""
	Perform pseudo–arc-length continuation starting from {u0 : tensor (B, Nx), lam0: (B, 1)} = sol_0.
	
	Key quantities:
	u0,       # (B, Nx)
	lam0,     # (B, 1)

	Args:
		ds: the arc-length step size
		model: cdft model considered
		# Q_flat: inference parameters tensor (B, Nq)

	Returns:
		sol_curve = [...,{sol_i},...]: list of dictionaries sol_i = {u_i : tensor (B, Nx), lam_i: scalar}. 
		Each item of the list represent a point of the curve: f(u, lam) = 0
	"""

	# # Q_flat = kwargs["Q_flat"]
	sol_curve = [sol_0]

	for k in range(continuation_steps):
		print()
		rho_, mu_ = _continuation_step(  
										u0 = sol_curve[-1]["rho"].to(model.sol["device"]),
										lam0 = sol_curve[-1]["mu"].to(model.sol["device"]),
										model = model,
										ds = ds,
										max_corrector_steps = max_corrector_steps,
										alpha = alpha,
										)
		
		omegaX_i, OmegaX_i = model.GetOmega(rho_)

		with torch.no_grad():
			sol_curve.append({
			"rho" :rho_.detach().cpu(), 
			"mu" : mu_.detach().cpu(),
			"omegaX" : omegaX_i.detach().cpu(),
			"OmegaX" : OmegaX_i.detach().cpu()/(2*model.mesh["L"]),
			"rhocoex_vl" : model.sol["rho_vl"],
			})
			torch.cuda.empty_cache()


			print("\step:",k, "\t\t" + model.eq_params["str_param"] + ":", mu_[0,0].item())
			print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
			print(f"Memory reserved:  {torch.cuda.memory_reserved() / 1e6:.2f} MB")

			each = 5
			if plotdir is not None and k%each==0:
				plotdir_norm = os.path.normpath(plotdir.rstrip(os.sep))
				os.makedirs(plotdir_norm, exist_ok=True)
				print("plotting in " + plotdir_norm + "...\n")
				rho_liq = model.sol["rho_vl"][1]
				rho_vap = model.sol["rho_vl"][0]
				plt.plot(model.mesh["x"].cpu().squeeze(), rho_.cpu().detach().squeeze(), "k", label=r"$\rho(x)$")
				plt.plot(model.mesh["x"].cpu(), (torch.ones_like(model.mesh["x"])*model.sol["rho_vl"][1]).cpu(), "r--", label=r"$\rho_l$")
				plt.plot(model.mesh["x"].cpu(), (torch.ones_like(model.mesh["x"])*model.sol["rho_vl"][0]).cpu(), "y--", label=r"$\rho_v$")
				# plt.plot(model.mesh["x"], torch.ones_like(model.mesh["x"])*model.sol["rhoB_vap"], "k--" ,label=r"$\rho_{b_0}$")
				plt.xlabel(r"$x$")
				plt.ylabel(r"$\rho$")
				plt.title(r"Density profile (step {})".format(k))
				plt.legend()
				plt.grid()
				plt.savefig(os.path.join(plotdir_norm, "rho_temp.png"))
				plt.clf()
				plt.close()

				# # omegaX, OmegaX, = [], []
				mean_rho, mu, c = [], [], []
				for i in range(len(sol_curve)):
					mu.append(sol_curve[i]["mu"][0,0].detach().item())
					mean_rho.append(torch.abs(sol_curve[i]["rho"]).mean(-1)[0].detach().item())
					# # # # omegaX_i, OmegaX_i = sol_curve[i]["omegaX"], sol_curve[i]["OmegaX"]
					# OmegaX.append(
						#  OmegaX_i[0])
					# omegaX.append(
						#  omegaX_i[0,:])
					c.append((i))
				scat = plt.scatter(mu, mean_rho, c=c, cmap="winter_r")
				plt.colorbar(scat, label="step")
				plt.xlabel(r"$\mu$")
				plt.ylabel(r"$\langle\rho\rangle$")
				plt.title(r"Chemical potential vs mean density")
				plt.grid()
				plt.savefig(os.path.join(plotdir_norm, "mu_meanrho.png"))
				plt.clf()
				plt.close()


				plt.plot(model.mesh["x"].cpu(), sol_curve[-1]["omegaX"][0,0,:])
				plt.xlabel(r"$x$")
				plt.ylabel(r"$\omega_X$")
				plt.title(r"Grand potential density (step {})".format(k))
				plt.grid()
				plt.savefig(os.path.join(plotdir_norm, "omX.png"))
				plt.clf()
				plt.close()

	
		if model.eq_params["ensemble"] == "NVT": 
			chem_pot = model.GetChemPot(num_par = mu_.detach(), rho_guess = rho_.detach())
			sol_curve[-1]["chem_pot"] = chem_pot.detach().cpu()
			print("Chemical potential:", chem_pot.squeeze().item())  
		### WARNING ###
		L_val = model.mesh["L"] if not torch.is_tensor(model.mesh["L"]) else model.mesh["L"].item()
		thresh = model.eq_params["target_density"] * (2 * L_val) + 0.1
		if mu_.squeeze().item() > thresh:
			break
	return sol_curve


def _update_residual_(lam, u, model,):
	model.eq_params["mu"] = lam.detach() 
	results = model.residual(   u.detach(),
								compute_Jac = True,
								detach_tensors = True,
								)
	

	res = results["res"].detach()  # (B, 1, Nx)
	dfdu = results["Jac_op"].detach()  # (B, 1, Nx, Nx)
	dfdlam = model.dfdmu(rho_h0 = u - res).detach()  # (B, 1, Nx, 1) 
	return res, dfdu, dfdlam


def _continuation_step(
	u0,
	lam0,
	model,
	ds,       # (,)
	max_corrector_steps,
	alpha,
	):
	"""
	Perform 'steps' of Keller pseudo–arc-length continuation starting from (u0, lam0).

	#WARNING#: it does not work with multiple batches. It only works for Ns=1.
	
	Key quantities:
	u0,       # (B, 1, Nx)
	lam0,     # (B, 1, 1)
	dfdu,     # (B, 1, Nx, Nx)
	dfdlam,   # (B, 1, Nx, 1)
	res,      # (B, 1, Nx)

	Args:
		x0 = (u0, lam0): initial solution
		X0 = (U0, Lam0): predictor solution
		Xk = (Uk, Lamk): intermediate newton solution
		ds: the arc-length step size

	Returns:
		x_new = (u_new, lam_new):   final newton solution. It solves: f(u_new, lam_new) = 0
	"""
	device = u0.device
	B = u0.shape[0]
	Nx = u0.shape[2]
	
	# 1. We need an initial tangent direction (dx/dlam).
	#    We approximate this from partial derivatives:
	#      F(u, lam) = 0  =>  (dF/du, du) + (dF/dlam, dlam) = 0
	# Solve  for TANGENT VECTOR
	# build the matrix A_t and know term b_t 
	"""      _            _           _ _
	#       |df/du|df/dlam |         | 0 |
	# A_t = |_____|_______ |,  b_t = | _ |
	#       |..1..|    1   |         | 1 |
	#       |_    |       _|         |_ _|
	"""
	#(B, 1, Nx), (B, 1, Nx, Nx), (B, 1, Nx, 1)
	res, dfdu, dfdlam = _update_residual_(lam0, 
											u0, 
											model, 
											# # Q_flat = kwargs["Q_flat"]
											)
		

	A_t_temp = torch.cat([dfdu, dfdlam], dim=-1)  # (B, 1, Nx, Nx+1)
	A_t = torch.cat([A_t_temp, 
					torch.ones(B,1, 1,Nx + 1, dtype=torch.double, device=device)
					], 
					dim=-2)  # (B, 1, Nx+1, Nx+1)
	
	b_t = torch.cat([
					torch.zeros((B, 1, Nx), dtype=torch.double, device=device),
					torch.ones((B, 1, 1), dtype=torch.double, device=device)
			], dim=-1)  # (B, 1, Nx+1)
	
	# Tangent vector
	tau = torch.linalg.solve(A_t, b_t)  # (B,1, Nx+1)
	if torch.isnan(tau).any() or torch.isinf(tau).any(): 
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nPredictor step failed: consider solving for the nullspace of the Jacobian\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		return None
	#scale = (tau[:,-1])
	#tau_n = tau / scale[:,None]
	tau_n = tau / torch.sqrt((tau**2).sum(-1)[...,None]) 

	# 2. PREDICTOR STEP
	#                         ds*tau
	# X0 = x0 + ds*tau     x0 ------> X0 
	#
	U0 = u0 		+ ds * tau_n[...,:-1]  # (B,1, Nx)
	Lam0 = lam0 	+ ds * tau_n[...,-1 ]  # (B,1, 1)
	


	# 3. CORRECTOR STEP ~ Newton loop
	# Eq.s:
	# dfdx * dX = - f;    
	# (tau, dX) = 0.
	#
	k=0
	dX = torch.ones_like(u0)
	while k < max_corrector_steps and torch.abs(dX).mean()/torch.abs(U0).mean().item() > 5e-8:
		# Compute residuals and jacobians at (U0, Lam0)
		res, dfdu, dfdlam = _update_residual_(
												Lam0, 
												U0, 
												model, 
												)
		

		# build the matrix A_p and know term b_p 
		"""      _            _                   _ _
		#       |df/du|df/dlam |                -|res|
		# A_c = |_____|_______ |,          b_c = | _ |
		#       | t_u | t_lam  |                 | 0 |
		#       |_    |       _| (X0)^k          |_ _| (X0)^k
		"""
		# First row block represent the problem F(X)_(N x N+1) = 0
		#
		# Second row block represent the ortogonality condition wrt the tangent vector tau
		
		A_c_top = torch.cat([dfdu, dfdlam], dim=-1)  # (B, 1, Nx, Nx+1)
		# A_c_down = torch.cat([torch.zeros((B, 1, Nx), dtype=torch.double), 
		#                       torch.ones((B, 1, 1), dtype=torch.double)], dim=2)  # (B, Nx, Nx+1)
		

		A_c = torch.cat([A_c_top,  # (B, 1, Nx, Nx+1)
						tau_n.unsqueeze(-2),  # (B, 1, 1, Nx+1)
										], 
									dim=-2)  # (B, 1, Nx+1, Nx+1)
		
		b_c = torch.cat([
							-res,  # (B,1, Nx)
							torch.zeros((B,1, 1), dtype=torch.double, device=device),
				], dim=-1)  # (B,1, Nx+1)

		dX = torch.linalg.solve(A_c, b_c)   # (B, 1, Nx+1)
		dU, dLam = dX[...,:-1], dX[...,-1]  # (B, 1, Nx), (B, 1)

		# Intermidiate newton step
		U1   = alpha * dU   + U0
		Lam1 = alpha * dLam + Lam0

		# Restart the loop
		U0   = U1
		Lam0 = Lam1
		k += 1

		del res, dfdu, dfdlam, A_c, b_c, dU, dLam
		torch.cuda.empty_cache()
	
	# print("|dU| =", torch.abs(dU).mean().item(),
	#       "\t|dLam| =", torch.abs(dLam).mean().item())
	print("|dX| =", torch.abs(dX).mean().item(), "|U| =", torch.abs(U1).mean().item())
	return U1, Lam1



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
