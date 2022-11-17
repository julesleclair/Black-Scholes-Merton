'''
Closed form solution of several options
'''

import numpy as np
import scipy.stats as stats


class BSM:

    '''
    --------------------------------------------
    Black-Scholes-Merton:
    --------------------------------------------

    Object that analytical solution of multiple options such as:
        - Vanilla European Option
        - Binary Options
        - Barrier Options
    ============================
    Input: Parameter Object

    ============================
    requirement:
        Numpy: import numpy as np
        Scipy: import scipy.stats as stats
    ============================
    How it works:
        obj.set_results(): Update Options Price and their Greeks
        obj.get_results(): Prints results
    ============================
    NB:
        If a specific value is required, use obj.r_****
    ===========================
:   Ressources:
    https://quantpie.co.uk/oup/oup_bsm_c_price_greeks_analysis.php
    Great website to check prices
    '''

    def __init__(self, param):

        self.param = param

        self.d1 = (np.log(self.param.stock/self.param.strike) \
                    + (self.param.rate - self.param.dividend \
                    + 0.5 * self.param.vol ** 2.0) \
                    * (self.param.tau)) / (self.param.vol * np.sqrt(self.param.tau))
        
        self.d2 = self.d1 - self.param.vol * np.sqrt(self.param.tau)

        # variable used for Barrier option pricing

        self.lambda_ = (self.param.rate - self.param.dividend
                        + 0.5 * self.param.vol**2.0) / self.param.vol**2.0

        #results: r_
        # TOdown_and_out: Maybe use down_and_outwn_and_inctionnary?
        # Option price
        self.r_eu_vanilla_c = 0
        self.r_eu_vanilla_p = 0

        # binarynary
        self.r_eu_binary_cash_or_nothing_c = 0
        self.r_eu_binary_cash_or_nothing_p = 0

        self.r_eu_binary_asset_or_nothing_c = 0
        self.r_eu_binary_asset_or_nothing_p = 0

        # Barrier

        self.r_eu_up_and_in_c = 0
        self.r_eu_up_and_in_p = 0

        self.r_eu_up_and_out_c = 0
        self.r_eu_up_and_out_p = 0

        self.r_eu_down_and_in_c = 0
        self.r_eu_down_and_in_p = 0

        self.r_eu_down_and_out_c = 0
        self.r_eu_down_and_out_p = 0

        # Greeks

        # Delta
        self.r_eu_vanilla_delta_c = 0
        self.r_eu_vanilla_delta_p = 0

        self.r_eu_binary_cash_or_nothing_delta_c = 0
        self.r_eu_binary_cash_or_nothing_delta_p = 0

        self.r_eu_binary_asset_or_nothing_delta_c = 0
        self.r_eu_binary_asset_or_nothing_delta_p = 0

        # Gamma
        self.r_eu_vanilla_gamma_c = 0
        self.r_eu_vanilla_gamma_p = 0

        self.r_eu_binary_cash_or_nothing_gamma_c = 0
        self.r_eu_binary_cash_or_nothing_gamma_p = 0

        self.r_eu_binary_asset_or_nothing_gamma_c = 0
        self.r_eu_binary_asset_or_nothing_gamma_p = 0

        # Vega
        self.r_eu_vanilla_vega_c = 0
        self.r_eu_vanilla_vega_p = 0

        self.r_eu_binary_cash_or_nothing_vega_c = 0
        self.r_eu_binary_cash_or_nothing_vega_p = 0

        self.r_eu_binary_asset_or_nothing_vega_c = 0
        self.r_eu_binary_asset_or_nothing_vega_p = 0

        # Theta
        self.r_eu_vanilla_theta_c = 0
        self.r_eu_vanilla_theta_p = 0

        self.r_eu_binary_cash_or_nothing_theta_c = 0
        self.r_eu_binary_cash_or_nothing_theta_p = 0

        self.r_eu_binary_asset_or_nothing_theta_c = 0
        self.r_eu_binary_asset_or_nothing_theta_p = 0

        # Rho
        self.r_eu_vanilla_rho_c = 0
        self.r_eu_vanilla_rho_p = 0

        self.r_eu_binary_cash_or_nothing_rho_c = 0
        self.r_eu_binary_cash_or_nothing_rho_p = 0

        self.r_eu_binary_asset_or_nothing_rho_c = 0
        self.r_eu_binary_asset_or_nothing_rho_p = 0

    def barrier_variables(self, H):
        """
        return parameter as define in John Hull's book: 
        Options, Futures, and Other Derivatives - 9th Edown_and_outwn_and_intion
        p626-628
        """
        y = np.log(H**2.0 / (self.param.stock * self.param.strike)) \
            / (self.param.vol * self.param.tau**0.5) \
            + self.lambda_ * self.param.vol * self.param.tau**0.5

        x1 = np.log(self.param.stock / H)  \
            / (self.param.vol * self.param.tau**0.5) \
            + self.lambda_ * self.param.vol * self.param.tau**0.5

        y1 = np.log(H / self.param.stock)  \
            / (self.param.vol * self.param.tau**0.5) \
            + self.lambda_ * self.param.vol * self.param.tau**0.5

        return y, x1, y1

    def _set_eu_vanilla(self):  # OK

        # Vanilla
        self.r_eu_vanilla_c = np.exp(-self.param.dividend * self.param.tau) \
            * self.param.stock * stats.norm.cdf(self.d1) \
            - np.exp(-self.param.rate*(self.param.tau)) \
            * self.param.strike * stats.norm.cdf(self.d2)
        self.r_eu_vanilla_p = - np.exp(-self.param.dividend * self.param.tau) \
            * self.param.stock * stats.norm.cdf(-self.d1) \
            + np.exp(-self.param.rate*(self.param.tau))\
            * self.param.strike * stats.norm.cdf(-self.d2)

        # binarynary
        # cash_or_nothing-or-nothing
        self.r_eu_binary_cash_or_nothing_c = np.exp(-self.param.rate*(self.param.tau)) \
            * stats.norm.cdf(self.d2)

        self.r_eu_binary_cash_or_nothing_p = np.exp(-self.param.rate*(self.param.tau)) \
            * stats.norm.cdf(-self.d2)

        # Asset-or-nothing
        self.r_eu_binary_asset_or_nothing_c = self.param.stock \
            * np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.cdf(self.d1)
        self.r_eu_binary_asset_or_nothing_p = self.param.stock \
            * np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.cdf(-self.d1)

    def _set_eu_delta(self):  # OK

        self.r_eu_vanilla_delta_c = np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.cdf(self.d1)
        self.r_eu_vanilla_delta_p = np.exp(-self.param.dividend * self.param.tau) \
            * (stats.norm.cdf(self.d1) - 1)

        self.r_eu_binary_cash_or_nothing_delta_c = np.exp(-self.param.rate * self.param.tau) \
            * stats.norm.pdf(self.d2) \
            / (self.param.stock * self.param.vol*np.sqrt(self.param.tau))
        self.r_eu_binary_cash_or_nothing_delta_p = -np.exp(-self.param.rate * self.param.tau) \
            * stats.norm.pdf(self.d2) \
            / (self.param.stock * self.param.vol*np.sqrt(self.param.tau))

        self.r_eu_binary_asset_or_nothing_delta_c = np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.pdf(self.d1) \
            / (self.param.vol*np.sqrt(self.param.tau)) \
            + np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.cdf(self.d1)

        self.r_eu_binary_asset_or_nothing_delta_p = -np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.pdf(self.d1) \
            / (self.param.vol*np.sqrt(self.param.tau)) \
            + np.exp(-self.param.dividend * self.param.tau) \
            * stats.norm.cdf(-self.d1)

    def _set_eu_gamma(self):

        self.r_eu_vanilla_gamma_c = 1/(self.param.stock * self.param.vol
                                       * np.sqrt(self.param.tau)) * stats.norm.pdf(self.d1) \
            * np.exp(-self.param.dividend * self.param.tau)
        self.r_eu_vanilla_gamma_p = self.r_eu_vanilla_gamma_c

        self.r_eu_binary_cash_or_nothing_gamma_c = -stats.norm.pdf(self.d2) * self.d1 * \
            np.exp(-self.param.rate * self.param.tau) / \
            (self.param.stock**2 * self.param.vol**2 * self.param.tau)
        self.r_eu_binary_cash_or_nothing_gamma_p = - \
            self.r_eu_binary_cash_or_nothing_gamma_c

        self.r_eu_binary_asset_or_nothing_gamma_c = -stats.norm.pdf(self.d1) * self.d2 \
            * np.exp(-self.param.dividend * self.param.tau) \
            / (self.param.stock * self.param.vol**2 * self.param.tau)
        self.r_eu_binary_asset_or_nothing_gamma_p = - \
            self.r_eu_binary_asset_or_nothing_gamma_c

    def _set_eu_vega(self):

        self.r_eu_vanilla_vega_c = self.param.stock * stats.norm.pdf(self.d1) \
            * np.sqrt(self.param.tau) \
            * np.exp(-self.param.dividend * self.param.tau)
        self.r_eu_vanilla_vega_p = self.r_eu_vanilla_vega_c

        self.r_eu_binary_cash_or_nothing_vega_c = -np.exp(-self.param.rate
                                                          * self.param.tau) \
            * stats.norm.pdf(self.d2) * self.d1 / self.param.vol

        self.r_eu_binary_cash_or_nothing_vega_p = - \
            self.r_eu_binary_cash_or_nothing_vega_c

        self.r_eu_binary_asset_or_nothing_vega_c = -self.param.stock \
            * np.exp(-self.param.dividend * self.param.tau)\
            * stats.norm.pdf(self.d1) * self.d2 \
            / self.param.vol

        self.r_eu_binary_asset_or_nothing_vega_p = - \
            self.r_eu_binary_asset_or_nothing_vega_c

    def _set_eu_theta(self):

        self.r_eu_vanilla_theta_c = (-1/(2*np.sqrt(self.param.tau))
                                     * self.param.stock * self.param.vol
                                     * stats.norm.pdf(self.d1)
                                     * np.exp(-self.param.dividend * self.param.tau)
                                     + self.param.dividend * self.param.stock
                                     * stats.norm.cdf(self.d1)
                                     * np.exp(-self.param.dividend * self.param.tau)
                                     - self.param.rate * self.param.strike
                                     * np.exp(-self.param.rate * self.param.tau)
                                     * stats.norm.cdf(self.d2))

        self.r_eu_vanilla_theta_p = (-1/(2*np.sqrt(self.param.tau))
                                     * self.param.stock * self.param.vol
                                     * stats.norm.pdf(self.d1)
                                     * np.exp(-self.param.dividend * self.param.tau)
                                     - self.param.dividend * self.param.stock
                                     * stats.norm.cdf(-self.d1)
                                     * np.exp(-self.param.dividend * self.param.tau)
                                     + self.param.rate * self.param.strike
                                     * np.exp(-self.param.rate * self.param.tau)
                                     * stats.norm.cdf(-self.d2))

        self.r_eu_binary_cash_or_nothing_theta_c = np.exp(-self.param.rate * self.param.tau) \
            * (stats.norm.pdf(self.d2)
               / (2 * self.param.vol * self.param.tau**1.5)
               * (np.log(self.param.stock/self.param.strike)
               - (self.param.rate - self.param.dividend
                  - 0.5 * self.param.vol**2.0) * self.param.tau)
               + self.param.rate * stats.norm.cdf(self.d2))

        self.r_eu_binary_cash_or_nothing_theta_p = np.exp(-self.param.rate * self.param.tau) \
            * (-stats.norm.pdf(self.d2)
               / (2 * self.param.vol * self.param.tau**1.5)
               * (np.log(self.param.stock/self.param.strike)
               - (self.param.rate - self.param.dividend
                  - 0.5 * self.param.vol**2.0) * self.param.tau)
               + self.param.rate * stats.norm.cdf(-self.d2))

        self.r_eu_binary_asset_or_nothing_theta_c = self.param.stock \
            * np.exp(-self.param.dividend * self.param.tau) \
            * (stats.norm.pdf(self.d1)
               / (2 * self.param.vol * self.param.tau**1.5)
               * (np.log(self.param.stock/self.param.strike)
               - (self.param.rate - self.param.dividend
                  + 0.5 * self.param.vol**2.0) * self.param.tau)
               + self.param.dividend * stats.norm.cdf(self.d1))

        self.r_eu_binary_asset_or_nothing_theta_p = self.param.stock \
            * np.exp(-self.param.dividend * self.param.tau) \
            * (-stats.norm.pdf(self.d1)
               / (2 * self.param.vol * self.param.tau**1.5)
               * (np.log(self.param.stock/self.param.strike)
               - (self.param.rate - self.param.dividend
                  + 0.5 * self.param.vol**2.0) * self.param.tau)
               + self.param.dividend * stats.norm.cdf(-self.d1))

    def _set_eu_rho(self):

        self.r_eu_vanilla_rho_c = self.param.strike * self.param.tau \
            * np.exp(-self.param.rate*self.param.tau) \
            * stats.norm.cdf(self.d2)
        self.r_eu_vanilla_rho_p = -self.param.strike * self.param.tau \
            * np.exp(-self.param.rate*self.param.tau) \
            * stats.norm.cdf(-self.d2)

        self.r_eu_binary_cash_or_nothing_rho_c = np.exp(-self.param.rate * self.param.tau) \
            * (np.sqrt(self.param.tau) * stats.norm.pdf(self.d2) / self.param.vol
               - self.param.tau * stats.norm.cdf(self.d2))
        self.r_eu_binary_cash_or_nothing_rho_p = np.exp(-self.param.rate * self.param.tau) \
            * (-np.sqrt(self.param.tau) * stats.norm.pdf(self.d2) / self.param.vol
               - self.param.tau * stats.norm.cdf(-self.d2))

        self.r_eu_binary_asset_or_nothing_rho_c = self.param.stock * np.sqrt(self.param.tau)\
            * np.exp(-self.param.dividend * self.param.tau)\
            * stats.norm.pdf(self.d1) / self.param.vol
        self.r_eu_binary_asset_or_nothing_rho_p = - \
            self.r_eu_binary_asset_or_nothing_rho_c

    def set_barrier(self, barrier=None):
        """
        Ressources: http://www.coggit.com/freetools
        to check price
        """
        if barrier == None:
            barrier = self.param.stock * 0.9

        y, x1, y1 = self.barrier_variables(barrier)

        if barrier <= self.param.strike:

            self.r_eu_down_and_in_c = self.param.stock \
                * np.exp(-self.param.dividend * self.param.tau) \
                * (barrier/self.param.stock)**(2 * self.lambda_) \
                * stats.norm.cdf(y) \
                - self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * (barrier / self.param.stock)**(2 * self.lambda_ - 2) \
                * stats.norm.cdf(y - self.param.vol * np.sqrt(self.param.tau))

            self.r_eu_down_and_out_c = self.r_eu_vanilla_c - self.r_eu_down_and_in_c

            self.r_eu_up_and_out_c = 0
            self.r_eu_up_and_in_c = self.r_eu_vanilla_c

            if self.param.stock >= barrier:
                self.r_eu_up_and_out_p = 0

            else:
                self.r_eu_up_and_out_p = -self.param.stock \
                    * stats.norm.cdf(-x1) \
                    * np.exp(-self.param.dividend * self.param.tau) \
                    + self.param.strike \
                    * np.exp(-self.param.rate * self.param.tau) \
                    * stats.norm.cdf(-x1 + self.param.vol * np.sqrt(self.param.tau)) \
                    + self.param.stock \
                    * np.exp(-self.param.dividend * self.param.tau)\
                    * (barrier / self.param.stock)**(2 * self.lambda_) \
                    * stats.norm.cdf(-y1) \
                    - self.param.strike \
                    * np.exp(-self.param.rate * self.param.tau) \
                    * (barrier / self.param.stock)**(2 * self.lambda_ - 2) \
                    * (stats.norm.cdf(-y1 + self.param.vol * np.sqrt(self.param.tau)))

            self.r_eu_up_and_in_p = self.r_eu_vanilla_p - self.r_eu_up_and_out_p

            self.r_eu_down_and_in_p = -self.param.stock \
                * stats.norm.cdf(-x1) \
                * np.exp(-self.param.dividend * self.param.tau) \
                + self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * stats.norm.cdf(-x1 + self.param.vol * np.sqrt(self.param.tau)) \
                + self.param.stock \
                * np.exp(-self.param.dividend * self.param.tau)\
                * (barrier / self.param.stock)**(2 * self.lambda_) \
                * (stats.norm.cdf(y) - stats.norm.cdf(y1)) \
                - self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * (barrier / self.param.stock)**(2*self.lambda_-2) \
                * (stats.norm.cdf(y - self.param.vol * np.sqrt(self.param.tau))
                   - stats.norm.cdf(y1 - self.param.vol * np.sqrt(self.param.tau)))

            self.r_eu_down_and_out_p = self.r_eu_vanilla_p - self.r_eu_down_and_in_p

        elif barrier >= self.param.strike:

            if self.param.stock <= barrier:
                self.r_eu_down_and_out_c = 0

            else:
                self.r_eu_down_and_out_c = self.param.stock \
                    * stats.norm.cdf(x1) \
                    * np.exp(-self.param.dividend * self.param.tau) \
                    - self.param.strike \
                    * np.exp(-self.param.rate * self.param.tau) \
                    * stats.norm.cdf(x1-self.param.vol * self.param.tau**0.5) \
                    - self.param.stock \
                    * np.exp(-self.param.dividend * self.param.tau)\
                    * (barrier / self.param.stock)**(2 * self.lambda_) \
                    * stats.norm.cdf(y1) \
                    + self.param.strike \
                    * np.exp(-self.param.rate * self.param.tau) \
                    * (barrier / self.param.stock)**(2 * self.lambda_ - 2) \
                    * (stats.norm.cdf(y1 - self.param.vol * self.param.tau**0.5))

            self.r_eu_down_and_in_c = self.r_eu_vanilla_c - self.r_eu_down_and_out_c

            self.r_eu_up_and_in_c = self.param.stock \
                * stats.norm.cdf(x1) \
                * np.exp(-self.param.dividend * self.param.tau) \
                - self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * stats.norm.cdf(x1-self.param.vol*np.sqrt(self.param.tau)) \
                - self.param.stock \
                * np.exp(-self.param.dividend * self.param.tau)\
                * (barrier / self.param.stock)**(2*self.lambda_) \
                * (stats.norm.cdf(-y) - stats.norm.cdf(-y1)) \
                + self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * (barrier / self.param.stock)**(2*self.lambda_-2) \
                * (stats.norm.cdf(-y + self.param.vol*np.sqrt(self.param.tau))
                   - stats.norm.cdf(-y1+self.param.vol*np.sqrt(self.param.tau)))

            self.r_eu_up_and_out_c = self.r_eu_vanilla_c - self.r_eu_up_and_in_c

            self.r_eu_up_and_in_p = -self.param.stock \
                * np.exp(-self.param.dividend * self.param.tau) \
                * (barrier/self.param.stock)**(2 * self.lambda_) \
                * stats.norm.cdf(-y) \
                + self.param.strike \
                * np.exp(-self.param.rate * self.param.tau) \
                * (barrier / self.param.stock)**(2 * self.lambda_ - 2) \
                * stats.norm.cdf(-y + self.param.vol * np.sqrt(self.param.tau))

            self.r_eu_up_and_out_p = self.r_eu_vanilla_p - self.r_eu_up_and_in_p

            self.r_eu_down_and_out_p = 0

            self.r_eu_down_and_in_p = self.r_eu_vanilla_p

    def set_results(self):
        self._set_eu_vanilla()
        self._set_eu_delta()
        self._set_eu_gamma()
        self._set_eu_vega()
        self._set_eu_theta()
        self._set_eu_rho()

    def get_results(self):
        """
        Print option price
        """
        count = 0
        print("{:<32} |  {:<12}  |  {:<12} ".format('Options', 'Call', 'Put'))
        print("_"*78)
        temp_c = 0
        for i in (vars(self)):
            if i[:2] == "r_":

                if count % 2 == 0:
                    temp_c = vars(self)[i]

                else:
                    print("{:<32} |  {:<12.7f}  |  {:<12.7f}".format(
                        i[5:-2], temp_c, vars(self)[i]))
                count += 1