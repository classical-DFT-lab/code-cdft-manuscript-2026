import torch


class REPULSIVE_FREE_ENERGY():
    def __init__(self):
        
        R = self.eq_params["R"]
        a = torch.tensor(6)/(torch.pi*(R)**3)
        self.eq_params["a"] = a[None] # (1,)


    # REPULSIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Repulsive Free Energy density
    # USE_DBH_DIAMETER: when True, use Barker-Henderson diameter scaling (a/dbh^3) in all cases
    #                   when False, use dbh only when USE_MODEL is False (legacy clean behavior)
    def phiR(self, rho):
        use_dbh = self.sol.get("USE_DBH_DIAMETER", False)
        if self.sol["USE_MODEL"]:
            if use_dbh:
                dbh = self.bh_diameter()  # (B,1)
                C = self.eq_params["pred_dnn"][...,3:4]
                a = self.eq_params["a"] * (1 + 1e-3*C) / dbh**3  # (B,1)
            else:
                a = self.eq_params["a"]    # (B,1)
        else:
            dbh = self.bh_diameter()  # (B,1)
            a = self.eq_params["a"][None,...] / dbh**3  # (1,1)
        beta = self.eq_params["beta"]  # (B,1)

        if self.sol["USE_MODEL"]:
            out_wda = self.wda_fn(rho)  # (B, 1, Nx)
            corr = out_wda
            # print(corr)
        else:
            corr = rho

        n = [
            corr / a[..., None] 
            ]  # [(B, 1, Nx)]

        # Compute list of inputs for phi:        
        phiR_0 = ( 
                    (a / beta)[...,None]  # (B, 1, 1)
                    *
                #n[0] * (3 - 2*n[0]) / (1 - n[0])**2
                n[0]**2 * (4 - 3*n[0]) / (1 - n[0])**2  # (B, 1, Nx)
                )  # (B, 1, Nx)

        # phiR_0 = ( 
        #              (1. / beta)[...,None]  # (B, 1, 1)
        #             *
        #            # n[0] * (3 - 2*n[0]) / (1 - n[0])**2
        #             n[0]**1 * (4 - 3*n[0]) / (1 - n[0])**2
        #         )  


        if self.sol["USE_MODEL"]:
            inputs = torch.cat([n[0], beta[...,None].expand(*(-1,)*len(beta.shape), rho.shape[-1]) ], dim=-2, )
            phi_corr = self.dnn_rep_fn(inputs)
            phiR = phiR_0*(1. + 0.01*phi_corr[:,0:1,:]) #+ phi_corr[:,1:2,:] 
        else:
            phiR = phiR_0
        return phiR  # (BS, 1, Nx)



# --> Repulsive Free Energy
    def FR(self, phiR):
        dx = self.mesh["dx"]
        FR = torch.einsum("...uj->...u", phiR) * dx  # (B,1)
        return FR


    def psi_rep(self, r):
        sigma_attr = 1.     #self.eq_params["sigma_attr"][..., None]  # (B, 1, 1)
        eps_attr = 1.     #self.eq_params["eps_attr"][..., None]        # (B, 1, 1)
        
        # Clamp x_abs to avoid division by near-zero values
        r_clamped = torch.clamp(r, min=1e-6)

        # Compute effective sigma and comp1
        comp1 = sigma_attr / r_clamped
        rep_exp = 12.0 #* (1.0 + 0.01 * exp_rep_corr)    # (B, 1, 1)
        attr_exp = 6.0 # * (1.0 + 0.01 * exp_attr_corr)  # (B, 1, 1)


        # Attractive-repulsive interaction
        psi_rep = (
         4*eps_attr * sigma_attr**3 *
            (
                (comp1 ** rep_exp) -
                (comp1 ** attr_exp) 
                )
            )
        return psi_rep

    def bh_diameter(self):
        grid = torch.linspace(0, 1**(1/6), 100, dtype=torch.double, device=self.sol["device"])
        dgrid = grid[1]-grid[0]
        u = self.psi_rep(grid)[None,None,:] # (B, 1, Ng)
        f = 1 - torch.exp(-self.eq_params["beta"][...,None]*u)
        return f.sum(-1) * dgrid
