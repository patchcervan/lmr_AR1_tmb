#include <TMB.hpp>                                // Links in the TMB libraries

template<class Type>
Type objective_function<Type>::operator() ()
{
    
    // data
    DATA_VECTOR( Y );           // response
    DATA_MATRIX( U );           // covariates random effects
    
    DATA_IMATRIX( series );     // start index and number of locations in each flight
    DATA_FACTOR( group );       // indicator for group
    DATA_INTEGER( J );          // number of groups
    
    int n = Y.rows();           // number of data points
    
    int L = U.cols();           // number of random parameters
    
    int nn = series.rows();     // number of series
    
    
    //RANDOM parameters;
    PARAMETER_MATRIX(B);
    
    
    // HYPERPARAMETERS for random effects
    PARAMETER_VECTOR(Mu_B);
    PARAMETER_VECTOR(logSigma_B);
    
    // Correlation parameter of the AR process
    PARAMETER(trphi);      // phi is common to all series
    
    // Transformed version of trphi. This will always be between -1 and 1.
    Type phi = 2.0 / (1.0 + exp(-trphi)) - 1.0;
    
    // Process variance
    PARAMETER(logSDp);
    
    // Process variance in linear scale for reports
    Type sigma_p = exp(logSDp);
    
    // Standard deviation for random effects in the original scale for reports
    vector<Type> Sigma_B = exp(logSigma_B);
    
    // Parameters for residuals
    PARAMETER(logSDe);
    
    // Standard deviations for residuals in the original scale for reports
    Type sigma = exp(logSDe);
    
    // Declare mu for the estimates
    vector<Type> mu(n);
    
    // Declare res for the residuals
    vector<Type> res(n);
    
    // auxiliary vector to subset the residuals of each series
    vector<Type> subRes;
    
    
    // objective function
    Type nll = 0.0;
    
    // Load AR functionality
    using namespace density;
    
    
    // Calculate expected values from the model
    for(int i = 0; i < n; ++i){
        
        mu[i] = 0.0;
        
        for(int l = 0; l < L; ++l){
            
            mu[i] += B(group[i],l) * U(i,l);
            
        }
        
    }
    
    // Calculate residuals
    res = Y - mu;
        
    
    // Neg. log-likelihood of data conditional on random effects
    nll = -sum(dnorm(Y, mu, exp(logSDe), true));
    
    
    // Neg. log-likelihood of random effects
    for(int l = 0; l < L; ++l){
        for(int j = 0; j < J; ++j){
            
            nll -= dnorm(B(j,l), Mu_B[l], exp(logSigma_B[l]), true);
            
        }
    }
    

    // Neg. log-likelihood of the AR1 process on mu. For each series
    for(int f = 0; f < nn; ++f){
        
        int s = series(f,0);
        int d = series(f,1);
        
        
        subRes = res.segment(s, d);
        
        nll += SCALE(AR1(phi), exp(logSDp))(subRes);
        
    }
    
    // Reporting
    ADREPORT( sigma );
    ADREPORT( Sigma_B );
    ADREPORT( phi );
    ADREPORT( sigma_p );
    return nll;
    
}
