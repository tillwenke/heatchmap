using LinearAlgebra


# Function to compute the radial basis function (RBF) kernel
# X_a: Vector of input values for the first point
# X_b: Vector of input values for the second point
# theta: Tuple of kernel hyperparameters (σ2_f, M)
# Returns: Float64 value representing the kernel evaluation
function rbf_kernel(X_a::Vector{<:Real}, X_b::Vector{<:Real}, theta::Tuple)::Float64
    M = diagm(theta[3])
    M_inverse = inv(M)

    k_ab = theta[1] * exp((-1) * transpose(X_a - X_b) * (M_inverse * M_inverse) * (X_a - X_b))
    
    if (X_a == X_b)
        k_ab += theta[2]
    end

    return k_ab
end


# Function to calculate the Ornstein-Uhlenbeck (OU) kernel
function ou_kernel(X_a::Vector{<:Real}, X_b::Vector{<:Real}, theta::Tuple)::Float64

    return k_ab
end


# Function to compute the kernel matrix
# X_a: Array of input values for the first set of points
# X_b: Array of input values for the second set of points
# kf: KernelFunction object with a defined kernel function
# theta: Tuple of kernel hyperparameters
# Returns: Kernel matrix G
function kernelmat(X_a::Array, X_b::Array, kernel::Function, theta::Tuple)
    # Initialize covariance matrix
    G = zeros(length(X_a), length(X_b))

    # Fill covariance matrix with kernel evaluations
    for i in 1:length(X_a)
        for j in 1:length(X_b)
            C[i, j] = kernel(X_a[i], X_b[j], theta)
        end
    end

    return G
end


# Define the GP regression model
struct GaussianProcess
    X::Array  # Training inputs
    y::Vector  # Training targets
    kernel::Function #kernel function
    theta::Tuple  # Kernel hyperparameters
    L::Matrix  # Cholesky decomposition of the covariance matrix
end


# Function to train a GP regression model
function train_gp(X::Array, y::Vector, kernel::Function, theta::Tuple)::GaussianProcess
    
end


# Function to make predictions with the GP regression model
# Returning the predicted mean and variance as output
function predict_gp(gp::GaussianProcess, X_pred::Array)
    

end


# Function to perform gradient descent optimization
function gradientdescent(gradient::Function,            # gradient function
                            x_0::Vector,                # initialization
                            Boundaries::Array;          # bounds on parameter space. First column is lower bound, second column is upper bound.
                            stepsize::Real = 0.0000001, 
                            eps::Real = 0.000005,       # convergence criteria
                            max_iter::Int = 5000        #maximum number of iterations
                            )
 
end


# Function to Compute log marginal likelihood of a GP model
function log_m_likelihood(gp::GaussianProcess)

    return lml
end


# Function to compute the gradients of the Radial Basis Function (RBF) kernel with respect to the parameters.
function grad_rbf(x_p, x_q, theta)::Vector

    return [partial_σ2_f, partial_σ2_n, partial_ls...]

end


# Function to compute the gradients of the log marginal likelihood with respect to the parameters of a Gaussian Process (GP).
function  grad_logmlik(gp::GaussianProcess)::Vector

    return grad
end



# Function to find the optimal values of the parameters (theta) for a Gaussian Process (GP) regression model by running gradient descent 
# on the negative log marginal likelihood.
function find_theta(X::Array, y::Vector, theta_init::Tuple)

    return theta_star
end