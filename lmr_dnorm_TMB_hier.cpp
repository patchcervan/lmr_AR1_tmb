#include <TMB.hpp>                                // Links in the TMB libraries

template<class Type>
Type objective_function<Type>::operator() ()
{
    
    // data
    DATA_VECTOR( Y );   // response
    //DATA_MATRIX( X );    // covariates fixed effects
    DATA_MATRIX( U );    // covariates random effects
    DATA_FACTOR( group );
    DATA_INTEGER( J );
    
    int n = Y.rows(); // number of data points
    
    //int K = X.cols();  // number of fixed parameters
    
    int L = U.cols();  // number of fixed parameters
    
    //vector<Type> factorLevels = group.attr(\"levels\");
    
    //int J = factorLevels.size();
    
    
    //FIXED parameters
    //PARAMETER_VECTOR(A);
    
    //RANDOM parameters;
    PARAMETER_MATRIX(B);
    
    // HYPERPARAMETERS for random effects
    PARAMETER_VECTOR(Mu_B);
    PARAMETER_VECTOR(logSigma_B);
    // Standard deviation for random effects in the original scale for reports
    vector<Type> Sigma_B = exp(logSigma_B);
    
    // Parameters for residuals
    PARAMETER(logSDe);
    
    // Standard deviations for residuals in the original scale for reports
    Type sigma = exp(logSDe);
    
    // objective function
    Type nll = 0.0;
    
    // Declare mu for the estimates
    vector<Type> mu(n);
    
    // Probability of data conditional on fixed and random effect values
    for(int i = 0; i < n; ++i){
        
        mu[i] = 0.0;
        
        mu[i] = A[i] * X(i,1);
        
        //for(int k = 0; k < K; ++k){
            
        //    
            
        //}
        
        for(int l = 0; l < L; ++l){
                
                A[i] += B(l,group[i]) * U(i,l);
    
        }
        
    }
    
    nll = -sum(dnorm(A, A, exp(logSD_A), true));    // (??)
    
    nll = -sum(dnorm(Y, mu, exp(logSDe), true));
    
    for(int l = 0; l < L; ++l){
        for(int j = 0; j < J; ++j){
            nll -= dnorm(B(j,l), Mu_B[l], exp(logSigma_B[l]), true);
        }
    }
    

    
    // Reporting
    ADREPORT( sigma );
    ADREPORT( Sigma_B );
    return nll;
    
}
