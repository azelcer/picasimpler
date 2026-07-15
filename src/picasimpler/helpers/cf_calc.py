import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

LIGHT_SPEED = 1

LAMBDA0 = 520

N1 = np.sqrt(1.77)
N3 = np.sqrt(2.13)
NUM_AP = 1.4

Z_START = 0
Z_STEP = 5
Z_END = Z_STEP + 300

class SimulQ():
    def __init__(
        self,
        lambda_0,
        n1,
        n3,
        na
    ):
        self.lambda_0 = lambda_0
        self.n1 = n1
        self.n3 = n3
        self.eps1 = self.n1**2
        self.eps3 = self.n3**2
        self.k0 = 2*np.pi/self.lambda_0
        self.k1 = self.k0 * self.n1
        self.k3 = self.k0 * self.n3
        self.prefac_s_3 = LIGHT_SPEED*self.eps3**(0.5)/(8*np.pi)
        self.na = na
        self.max_angle = np.arcsin(self.na/self.n3)
    
        # z range for all simulations
        self.z_arr = np.arange(Z_START, Z_END, Z_STEP)
        self.precomp_diss_powers()

    def precomp_diss_powers(self):
        self.p_t_perp_arr = np.zeros(len(self.z_arr))
        self.p_t_parall_arr = np.zeros(len(self.z_arr))
        for z_idx, zz in enumerate(self.z_arr):
            #print(f"Calculating dissipated powers for z={zz} nm")
            self.p_t_perp_arr[z_idx] = self.p_t_perp(zz)
            self.p_t_parall_arr[z_idx] = self.p_t_parall(zz)
        self.eta_arr = self.p_t_parall_arr/self.p_t_perp_arr
        
        self.p_t_perp_interp = interp1d(self.z_arr, self.p_t_perp_arr)
        self.p_t_parall_interp = interp1d(self.z_arr, self.p_t_parall_arr)
        self.eta_interp = interp1d(self.z_arr, self.eta_arr)
        
    def plot_diss_powers(self):
        plt.plot(self.z_arr, self.p_t_perp_arr/np.max(self.p_t_perp_arr), label=r"$P_{T}^{\perp}$")
        plt.plot(self.z_arr, self.p_t_parall_arr/np.max(self.p_t_perp_arr), label=r"$P_{T}^{\parallel}$")
        plt.plot(self.z_arr, self.eta_arr, label=r"$\eta$")
        plt.ylim([0, 1.5])
        plt.legend()
        plt.show()
        
    def calc_q(self):
        self.q_perp_arr = np.zeros(len(self.z_arr))
        self.q_parall_arr = np.zeros(len(self.z_arr))
        self.q_arr = np.zeros(len(self.z_arr))
        for zz_idx, zz in enumerate(self.z_arr):
            #print(f"Calculating collection efficiency for z={zz} nm")
            self.q_perp_arr[zz_idx] = self.q_perp(zz)
            self.q_parall_arr[zz_idx] = self.q_parall(zz)
        
        self.q_perp_interp = interp1d(self.z_arr, self.q_perp_arr)
        self.q_parall_interp = interp1d(self.z_arr, self.q_parall_arr)
        
        for zz_idx, zz in enumerate(self.z_arr):
            self.q_arr[zz_idx] = self.q_final(zz)
            
        self.q_interp = interp1d(self.z_arr, self.q_arr, fill_value='extrapolate')
        return self.q_interp
        
    def plot_q(self):
        plt.plot(self.z_arr, self.q_perp_arr, label=r"$Q^{\perp}$")
        plt.plot(self.z_arr, self.q_parall_arr, label=r"$Q^{\parallel}$")
        plt.plot(self.z_arr, self.q_arr, label="Q")
        plt.xlim([0, Z_END])
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    @staticmethod
    def compl_sqrt_one_minxsq(xx):
        """
        square root of 1 - x**2, extended to complex numbers
        """
        return np.sqrt(1 - complex(xx)**2)

    def alpha(self, theta):
        return self.compl_sqrt_one_minxsq((N3/N1)*np.sin(theta))

    def refl_fac_p(self, v):
        """
        generalized Fresnel coeff for reflected light, p polarization
        """
        costheta_gen = self.compl_sqrt_one_minxsq(v)
        return (N3*costheta_gen - N1*self.compl_sqrt_one_minxsq((N1/N3)*v))/(N1*self.compl_sqrt_one_minxsq((N1/N3)*v) + N3*costheta_gen)

    def trans_fac_p(self, v):
        """
        generalized Fresnel coeff for transmitted light, p polarization
        """
        return (1 + self.refl_fac_p(v))*(N1/N3)

    def refl_fac_s(self, v):
        costheta_gen = self.compl_sqrt_one_minxsq(v)
        return (N1*costheta_gen - N3*self.compl_sqrt_one_minxsq((N1/N3)*v))/(N1*costheta_gen + N3*self.compl_sqrt_one_minxsq((N1/N3)*v))

    def trans_fac_s(self, v):
        return 1 + self.refl_fac_s(v)

    def p_t_perp_integrand(self, v, zz):
        """
        Eq 38
        """
        return (v/self.compl_sqrt_one_minxsq(v)*(v**2*(1 + self.refl_fac_p(v)*np.exp(1j*2*self.k1*zz*self.compl_sqrt_one_minxsq(v))))).real

    def p_t_perp(self, zz):
        """
        Eq 38
        """
        integral_0to1, _ = quad(self.p_t_perp_integrand, 0, 1, args=(zz))
        integral_1toinf, _ = quad(self.p_t_perp_integrand, 1, np.inf, args=(zz))
        integral_0toinf = integral_0to1 + integral_1toinf
        return LIGHT_SPEED*self.k1**4/(2*self.eps1**(3/2))*integral_0toinf
        
    def p_t_parall_integrand(self, v, zz):
        """
        Eq 38
        """
        return (v/self.compl_sqrt_one_minxsq(v)*((1 + self.refl_fac_s(v)*np.exp(1j*2*self.k1*zz*self.compl_sqrt_one_minxsq(v))) + (1 - v**2)*(1 - self.refl_fac_p(v)*np.exp(1j*2*self.k1*zz*self.compl_sqrt_one_minxsq(v))))/2).real
        
    def p_t_parall(self, zz):
        """
        Eq 38
        """
        integral_0to1, _ = quad(self.p_t_parall_integrand, 0, 1, args=(zz))
        integral_1toinf, _ = quad(self.p_t_parall_integrand, 1, np.inf, args=(zz))
        integral_0toinf = integral_0to1 + integral_1toinf
        return LIGHT_SPEED*self.k1**4/(2*self.eps1**(3/2))*integral_0toinf

    def e_mu_p_sq(self, zz, theta):
        """
        Eq 35b
        """
        return self.k3**4*np.cos(theta)**2*abs(self.trans_fac_p((N3/N1)*np.sin(theta))*np.exp(1j*self.k3*self.alpha(theta)*zz))**2/(self.eps1*self.eps3)

    def e_mu_s_sq(self, zz, theta):
        """
        Eq 36b
        """
        return self.k3**4*np.cos(theta)**2*abs(self.trans_fac_s((N3/N1)*np.sin(theta))*np.exp(1j*self.k1*self.alpha(theta)*zz))**2/(self.eps1*self.eps3*abs(self.alpha(theta))**2)

    def e_mu_z_sq(self, zz, theta):
        """
        Eq 37b
        """
        return self.k3**4*np.cos(theta)**2*np.sin(theta)**2*abs(self.trans_fac_p((N3/N1)*np.sin(theta))*np.exp(1j*self.k1*self.alpha(theta)*zz))**2/(self.eps1**2*abs(self.alpha(theta))**2)

    def s_hat_perp(self, theta, zz):
        """
        Eq 43
        """
        return self.prefac_s_3*self.e_mu_z_sq(zz, theta)/self.p_t_perp_interp(zz)

    def s_hat_perp_timessin(self, theta, zz):
        return np.sin(theta)*self.s_hat_perp(theta, zz)

    def s_hat_parall(self, theta, zz):
        """
        Eq 44
        """
        return self.prefac_s_3/(2*self.p_t_parall_interp(zz))*(self.e_mu_p_sq(zz, theta) + self.e_mu_s_sq(zz, theta))

    def s_hat_parall_timessin(self, theta, zz):
        return np.sin(theta)*self.s_hat_parall(theta, zz)

    def q_perp(self, zz):
        """
        Eq 49
        """    
        return 2*np.pi*quad(self.s_hat_perp_timessin, 0, self.max_angle, args=(zz))[0]
        
    def q_parall(self, zz):
        """
        Eq 49
        """
        return 2*np.pi*quad(self.s_hat_parall_timessin, 0, self.max_angle, args=(zz))[0]

    def perp_dir_weight(self, thetap, zz):
        return 0.5*np.sin(thetap)/(1 + self.eta_interp(zz)*np.tan(thetap)**2)

    def parall_dir_weight(self, thetap, zz):
        return 0.5*np.sin(thetap)/(1 + (1 / (self.eta_interp(zz)*np.tan(thetap)**2)))

    def q_final(self, zz):
        perp_term = self.q_perp_interp(zz)*quad(self.perp_dir_weight, 0, np.pi, args=(zz))[0]
        parall_term = self.q_parall_interp(zz)*quad(self.parall_dir_weight, 0, np.pi, args=(zz))[0]
        return perp_term + parall_term

if __name__=="__main__":
    simul_q = SimulQ(LAMBDA0, N1, N3, NUM_AP)
    simul_q.plot_diss_powers()
    simul_q.calc_q()
    simul_q.plot_q()
    
        