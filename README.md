# Black-Scholes-Merton
Closed form solution of the Black-Scholes-Merton formula for various types of **european Call/Put** options. 


    - Vanilla:
        - Greeks:
            - Delta
            - Gamma
            - Tau
            - Rho
            - Vega

    - Binary Cash or nothing
        - Greeks:
            - Delta
            - Gamma
            - Tau
            - Rho
            - Vega

    - Binary Asset or nothing
        - Greeks:
            - Delta
            - Gamma
            - Tau
            - Rho
            - Vega

    - Barriers
        - Knock-in
            - Up and In
            - Down and In
        - Knock-out
            - Up and Out
            - Down and out

Files:
    - option_param.py
        Object: story option parameter such as Initial price, strike, vol, etc...
    - bsm.py
        Object: Calculate Option price and their greeks (if applicable) for various options. 

    - Image: 
        Folder: contains surface plot
    
    - Black_Scholes_Merton.ipynb
        Jupyter notebook: Shows how to use the object BSM.