# Import ML libraries
import Flux
import NNlib
import Zygote
import SimpleChains

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase
import Distributions
import Distances

@doc raw"""
    `VAE`

Structure containing the components of a variational autoencoder (VAE).

NOTE: This method was created before I discovered the basic `Flux.identity`
function to create linear layers. Thus, its use will be deprecated

# Fields
- `encoder::Flux.Chain`: neural network that takes the input and passes it
   through hidden layers.
- `µ::Flux.Dense`: Single layers that map from the encoder to the mean (`µ`) of
the latent variables distributions and 
- `logσ::Flux.Dense`: Single layers that map from the encoder to the log of the
   standard deviation (`logσ`) of the latent variables distributions.
- `decoder`: Neural network that takes the latent variables and tries to
   reconstruct the original input.
"""
mutable struct VAE
    encoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
    decoder::Flux.Chain
end

@doc raw"""
    `vae_init(
        n_input, 
        n_latent, latent_activation, 
        encoder, encoder_activation,
        decoder, decoder_activation
    )`

Function to initialize a variational autoencoder neural network with `Flux.jl`.

# Arguments
- `n_input::Int`: Dimension of input space.
- `n_latent::Int`: Dimension of latent space
- `latent_activation::Function`: Activation function coming in of the latent
  space layer.
- `output_activation::Function`: Activation function on the output layer
- `encoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the encoder network (one layer per entry).
- `encoder_activation::Vector`: Array containing the activation function for the
  encoder hidden layers. If `nothing` is given, no activation function is
  assigned to the layer. NOTE: length(encoder) must match
  length(encoder_activation).
- `decoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the decoder network (one layer per entry).
- `decoder_activation::Vector`: Array containing the activation function for the
  decoder hidden layers. If `nothing` is given, no activation function is
  assigned to the layer. NOTE: length(encoder) must match
  length(encoder_activation).

## Optional arguments
- `init::Function=Flux.glorot_uniform`: Function to initialize network
parameters.

# Returns
- a `struct` of type `VAE`
"""
function vae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{Flux.Dense}(undef, length(encoder))

    # Loop through layers   
    for i = 1:length(encoder)
        # Check if it is the first layer
        if i == 1
            # Set first layer from input to encoder with activation
            Encoder[i] = Flux.Dense(
                n_input => encoder[i], encoder_activation[i]; init=init
            )
        else
            # Set middle layers from input to encoder with activation
            Encoder[i] = Flux.Dense(
                encoder[i-1] => encoder[i], encoder_activation[i]; init=init
            )
        end # if
    end # for

    # Define layer that maps from encoder to latent space with activation
    Latent_µ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )
    Latent_logσ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )

    # Initialize list with decoder layers
    Decoder = Array{Flux.Dense}(undef, length(decoder) + 1)

    # Add first layer from latent space to decoder
    Decoder[1] = Flux.Dense(
        n_latent => decoder[1], decoder_activation[1]; init=init
    )

    # Add last layer from decoder to output
    Decoder[end] = Flux.Dense(
        decoder[end] => n_input, output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder) > 1
        # Loop through middle layers
        for i = 2:length(decoder)
            # Set middle layers of decoder
            Decoder[i] = Flux.Dense(
                decoder[i-1] => decoder[i], decoder_activation[i]; init=init
            )
        end # for
    end # if

    # Compile encoder and decoder into single chain
    return VAE(
        Flux.Chain(Encoder...), Latent_µ, Latent_logσ, Flux.Chain(Decoder...)
    )
end # function

@doc raw"""
    `vae_reconstruct(input, vae)`
This function performs three steps:
1. passes an input `x` through the `encoder`, 
2. samples the latent variable by using the reparametrization trick,
3. reconstructs the input from the latent variables using the `decoder`.

# Arguments
- `input::Vector`: Input to the neural network.
- `vae::VAE`: Variational autoencoder struct with all components.

## Optional Arguments
- `latent::Bool=false`: Boolean indicating if the latent variables should be
  returned as part of the output or not.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
    to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
    the input when mapped to the latent space.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
    autoencoder. Note: This last point depends on a random sampling step, thus
    it will change every time.
"""
function vae_reconstruct(
    input::AbstractVector{Float32},
    vae::VAE;
    latent::Bool=false
)
    # 1. Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(input)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(input)

    # 2. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
    ) .* exp.(logσ)

    # 3. Run sampled latent variables through decoder and return values
    if latent
        return µ, logσ, z, vae.decoder(z)
    else
        return µ, logσ, vae.decoder(z)
    end # if
end # function

@doc raw"""
    `vae_loss(x, vae; σ, β, reconstruct, n_samples)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || P(z)),

where the minimization is taken over the functions f, g, and h. f(z) encodes the
function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f(x), σI).

g and h define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g(x), h(x)).

# Arguments
- `x::Vector`: Input to the neural network.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `n_samples::Int`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function vae_loss(
    x::AbstractVector{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    reconstruct::Function=vae_reconstruct,
    n_samples::Int=1
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x .- x̂) .^ 2)
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + β * kl_qₓ_p

end #function

@doc raw"""
    `vae_loss(x, x_true, vae; σ, β, reconstruct, n_samples)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || P(z)),

where the minimization is taken over the functions f, g, and h. f(z) encodes the
function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f(x), σI).

g and h define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g(x), h(x)).

# Arguments
- `x::Vector`: Input to the neural network.
- `x_true::Vector`: True input against which to compare autoencoder
  reconstruction.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `n_samples::Int`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function vae_loss(
    x::AbstractVector{Float32},
    x_true::AbstractVector{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    reconstruct::Function=vae_reconstruct,
    n_samples::Int=1
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + β * kl_qₓ_p

end #function

@doc raw"""
    `vae_kl_div(x, vae)`
Function to compute the KL divergence between the approximate encoder qₓ(z) and
the latent variable prior distribution P(z). Since we assume
        P(z) = Normal(0̲, 1̲),
and
        qₓ(z) = Normal(f(x̲), σI̲̲),
the KL divergence has a closed form

# Arguments
- `x::Vector`: Input to the neural network.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.        

# Returns
Dₖₗ(qₓ(z)||P(z))
"""
function vae_kl_div(x::AbstractVector{Float32}, vae::VAE)
    # Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(x)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(x)

    return sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)
end # function

@doc raw"""
    `vae_train!(loss, vae, data, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `loss::Function`: The loss function that defines the variational autoencoder.
  The gradient of this function (∇loss) will be automatically computed using the
  `Zygote.jl` library.
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `data::Matrix{Float32}`: Matrix containing the data on which to evaluate the
  loss function. NOTE: Every column should represent a single input.
- `opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the autoencoder parameters. This should be fed already with the
  corresponding parametres. For example, one could feed: ⋅ Flux.AMSGrad(η)

## Optional arguments
- `kwargs::NamedTuple`: Tuple containing arguments for the loss function.
    For `vae_loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
    through the autoencoder.
    - `n_samples::Int`: Number of samples to take from the latent space when
    computing ⟨logP(x|z)⟩.
"""
function vae_train!(
    loss::Function,
    vae::VAE,
    data::Matrix{Float32},
    opt::Flux.Optimise.AbstractOptimiser;
    kwargs...
)
    # Extract parameters
    params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)

    # Perform computation for first datum.
    # NOTE: This is to properly initialize the object on which the gradient will
    # be evaluated. There's probably better ways to do this, but this works.

    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    loss_, back_ = Zygote.pullback(params) do
        loss(data[:, 1], vae; kwargs...)
    end # do
    # Having computed the pullback, we compute the loss function gradient
    ∇loss_ = back_(one(loss_))

    # Loop through the rest of the datasets data
    for (i, d) in enumerate(eachcol(data[:, 2:end]))
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        loss_, back_ = Zygote.pullback(params) do
            loss(d, vae; kwargs...)
        end # do
        # Having computed the pullback, we compute the loss function gradient
        ∇loss_ .+= back_(one(loss_))
    end # for

    # Update the network parameters averaging gradient from all datasets
    Flux.Optimise.update!(opt, params, ∇loss_ ./ size(data, 2))
end # function

@doc raw"""
    `vae_train!(loss, vae, data, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function. For this method, the data consists of a `Array{Float32,
3}` object, where the third dimension contains both the noisy data and the
"real" value against which to compare the reconstruction.

# Arguments
- `loss::Function`: The loss function that defines the variational autoencoder.
  The gradient of this function (∇loss) will be automatically computed using the
  `Zygote.jl` library.
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `data::Array{Float32, 3}`: Matrix containing the data on which to evaluate the
  loss function. NOTE: Every column should represent a single input.
- `opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the autoencoder parameters. This should be fed already with the
  corresponding parametres. For example, one could feed: ⋅ Flux.AMSGrad(η)

## Optional arguments
- `kwargs::NamedTuple`: Tuple containing arguments for the loss function. For
    `vae_loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `reconstruct::Function`: Function that reconstructs the input x̂ by
    passing it through the autoencoder.
    - `n_samples::Int`: Number of samples to take from the latent space when
    computing ⟨logP(x|z)⟩.
"""
function vae_train!(
    loss::Function,
    vae::VAE,
    data::Array{Float32,3},
    opt::Flux.Optimise.AbstractOptimiser;
    kwargs...
)
    # Extract parameters
    params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)

    # Split data and real value
    data_noise = data[:, :, 1]
    data_true = data[:, :, 2]
    # Perform computation for first datum.
    # NOTE: This is to properly initialize the object on which the gradient will
    # be evaluated. There's probably better ways to do this, but this works.

    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    loss_, back_ = Zygote.pullback(params) do
        loss(data_noise[:, 1], data_true[:, 1], vae; kwargs...)
    end # do
    # Having computed the pullback, we compute the loss function gradient
    ∇loss_ = back_(one(loss_))

    # Loop through the rest of the datasets data
    for i = 2:size(data_noise, 2)
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        loss_, back_ = Zygote.pullback(params) do
            loss(data_noise[:, i], data_true[:, i], vae; kwargs...)
        end # do
        # Having computed the pullback, we compute the loss function gradient
        ∇loss_ .+= back_(one(loss_))
    end # for

    # Update the network parameters averaging gradient from all datasets
    Flux.Optimise.update!(opt, params, ∇loss_ ./ size(data, 2))
end # function


@doc raw"""
    `vae_mse_boots(vae, data, n_samples)`

Function to compute a bootstrap sample of the mean squared error for a
variational autoencoder.

# Arguments
- `vae::VAE`: Struct containing the elements of a variational autoencoder.
- `data::Matrix{Float32}`: Matrix containing the data on which to evaluate the
  loss function. NOTE: Every column should represent a single input.
- `n_samples::Int`: Number of bootstrap samples to generate.

# Returns
- `MSE_boots:Array{Float32}`: Mean squared error bootstrap samples.
"""
function vae_mse_boots(vae::VAE, data::Matrix{Float32}, n_samples::Int)
    # Initialize array to save bootstrap samples
    boots_samples = Array{Float32}(undef, n_samples)

    # Loop through samples
    for i = 1:n_samples
        # Compute mean squared error for sample i
        boots_samples[i] = StatsBase.mean([
            Flux.mse(d, vae_reconstruct(d, vae)[end]) for d in eachcol(data)
        ])
    end # for

    return boots_samples
end # function

@doc raw"""
    `vae_annealing(epoch, n_epoch, n_cycles; frac=0.5, βmax=1, βmin=0)`

Function that computes the value of the annealing parameter β as a function of
the epoch number according to the cyclical annealing strategy.

# Arguments
- `epoch::Int`: Epoch on which to evaluate the value of the annealing parameter.
- `n_epoch::Int`: Number of epochs that will be run to train the VAE.
- `n_cycles::Int`: Number of annealing cycles to be fit within the number of
  epochs.

## Optional Arguments
- `frac::AbstractFloat= 0.5`: Fraction of the cycle in which the annealing
  parameter β will increase from the minimum to the maximum value.
- `βmax::AbstractFloat=1.0`: Maximum value that the annealing parameter can
  reach.
- `βmin::AbstractFloat=0.0`: Minimum value that the annealing parameter can
  reach.

# Returns
- `β::Float32`: Value of the annealing parameter.
"""
function vae_cyc_annealing(
    epoch::Int,
    n_epoch::Int,
    n_cycles::Int;
    frac::AbstractFloat=0.5,
    βmax::AbstractFloat=1.0,
    βmin::AbstractFloat=0.0
)
    # Define variable τ that will serve to define the value of β
    τ = mod(epoch - 1, ceil(n_epoch / n_cycles)) / (n_epoch / n_cycles)

    # Compute and return the value of β
    if τ ≤ frac
        return convert(Float32, (βmax - βmin) * τ / frac + βmin)
    else
        return convert(Float32, βmax)
    end # if
end # function

@doc raw"""
    `gaussian_kernel(x, y)`
Function to compute the Gaussian Kernel between two vectors `x` and `y`, defined
as
    k(x, y) = exp(-||x - y ||² / ρ²)

# Arguments
- `x::Matrix{Float32}`: First array in kernel
- `y::Matrix{Float32}`: Second array in kernel

## Optional Arguments
- `ρ::Float32`: Kernel amplitude hyperparameter.

# Returns
k(x, y) = exp(-||x - y ||² / ρ²)
"""
function gaussian_kernel(
    x::Matrix{Float32}, y::Matrix{Float32}; ρ::Float32=1.0f0
)
    # return Gaussian kernel
    return exp.(
        -Distances.pairwise(
            Distances.SqEuclidean(), x, y
        ) ./ ρ^2 ./ size(x, 1)
    )
end # function

@doc raw"""
    `mmd_div(x, y)`
Function to compute the MMD divergence between two vectors `x` and `y`, defined
as
    D(x, y) = k(x, x) - 2 k(x, y) + k(y, y),
where k(⋅, ⋅) is any positive definite kernel.

# Arguments
- `x::Matrix{Float32}`: First array in kernel
- `y::Matrix{Float32}`: Second array in kernel

## Optional argument
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kwargs::NamedTuple`: Tuple containing arguments for the Kernel function.
"""
function mmd_div(
    x::Matrix{Float32},
    y::Matrix{Float32};
    kernel::Function=gaussian_kernel,
    kwargs...
)
    # Compute and return MMD divergence
    return StatsBase.mean(kernel(x, x; kwargs...)) +
           StatsBase.mean(kernel(y, y; kwargs...)) -
           2 * StatsBase.mean(kernel(x, y; kwargs...))
end # function

@doc raw"""
    `logP_mmd_ratio(x, vae; σ, n_latent_samples)`
Function to compute the ratio between the log probability ⟨log P(x|z)⟩ and the
MMD divergence MMD-D(qᵩ(z|x)||P(z)).

NOTE: This function is useful to define the value of the hyperparameter λ for
the infoVAE training.

# Arguments
- `x::Matrix{Float32}`: Data to train the infoVAE.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional Arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `n_latent_samples::Int`: Number of samples to take from the latent space prior
  P(z) when computing the MMD divergence.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
abs(⟨log P(x|z)⟩ / MMD-D(qᵩ(z|x)||P(z)))
"""
function logP_mmd_ratio(
    x::Matrix{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    n_latent_samples::Int=100,
    reconstruct=vae_reconstruct,
    kernel=gaussian_kernel,
    kernel_kwargs...
)
    # Initialize value to save log probability
    logP_x_z = 0.0f0
    # Initialize value to save MMD divergence
    mmd_q_p = 0.0f0

    # Loop through dataset
    for (i, x_datum) in enumerate(eachcol(x))
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x_datum, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x_datum) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x_datum .- x̂) .^ 2)

        # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
        # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
        mmd_q_p += mmd_div(
            # Sample latent variables from decoder qᵩ(z|x) ~ Normal(µ,
            # exp(2logσ)⋅I)
            µ .+ (Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
            ).*exp.(logσ))[:, :],
            # Sample latent variables from prior P(z) ~ Normal(0, 1)
            Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0),
                length(µ),
                n_latent_samples,
            );
            kernel=kernel,
            kernel_kwargs...
        )
    end # for

    # Return ratio of quantities
    return convert(Float32, abs(logP_x_z / mmd_q_p))
end # function

@doc raw"""
    `mmd_loss(x, vae; σ, λ, α, reconstruct, n_samples, kernel_kwargs...)`

Loss function for the Maximum-Mean Discrepancy variational autoencoder. The loss
function is defined as

loss = argmin -⟨⟨log P(x|z)⟩⟩ + (1 - α) ⟨Dₖₗ(qᵩ(z | x) || P(z))⟩ + 
              (λ + α - 1) Dₖₗ(qᵩ(z) || P(z)),

where the minimization is taken over the functions f, g, and h. f(z) encodes the
function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f(x), σI).

g and h define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g(x), h(x)).

# Arguments
- `x::Vector`: Input to the neural network.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `λ::Float32=1`: 
- `α::Float32=1`: Related to the annealing inverse temperature for the
  KL-divergence term.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `n_samples::Int`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩.
- `n_latent_samples::Int`: Number of samples to take from the latent space prior
  P(z) when computing the MMD divergence.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function mmd_loss(
    x::AbstractVector{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    λ::Float32=1.0f0,
    α::Float32=0.0f0,
    reconstruct::Function=vae_reconstruct,
    kernel::Function=gaussian_kernel,
    n_samples::Int=1,
    n_latent_samples::Int=50,
    kernel_kwargs...
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0
    # Initialize value to save MMD divergence
    mmd_q_p = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

        # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
        # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
        mmd_q_p += mmd_div(
            # Sample the decoder qᵩ(z | x) ~ Normal(µ, exp(2logσ)⋅I)
            µ .+ (Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
            ).*exp.(logσ))[:, :],
            # Sample the prior probability P(z) ~ Normal(0, 1)
            Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0),
                length(µ),
                n_latent_samples,
            );
            kernel=kernel,
            kernel_kwargs...
        )
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + (1 - α) * kl_qₓ_p +
           (λ + α - 1) * mmd_q_p / n_samples
end # function

@doc raw"""
    `infomax_loss(x, vae, mlp; σ, β, α, reconstruct, n_samples)`

Loss function for the infoMax variational autoencoder. The loss function is
defined as

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables.

# Arguments
- `x::Vector`: Input to the neural network.
- `x_shuffle::Vector`: Shuffled input to the neural network needed to compute
  the mutual information term. This term is used to obtain an encoding
  `z_shuffle` that represents a random sample from the marginal P(z).
- `vae::VAE`: Struct containing the elements of the variational autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
  between input and output.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `α::Float32=1`: Annealing inverse temperature for the mutual information term.
- `n_samples::Int=1`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
  over multiple latent-space samples due to the variability of each sample. In
  practice, most algorithms keep this at 1.

# Returns
- `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
  when compared with reconstructed output `x̂`. This is used by the training
  algorithms to improve the reconstruction.
"""
function infomax_loss(
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32},
    vae::VAE,
    mlp::Flux.Chain;
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
        # Run shuffled input through reconstruction function
        _, _, z_shuffle, _ = vae_reconstruct(
            x_shuffle, vae; latent=true
        )

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

        # Run input and latent variables through mutual information MLP
        I_xz = first(mlp([x; z]))
        # Run input and PERMUTED latent variables through mutual info MLP
        I_xz_perm = first(mlp([x; z_shuffle]))
        # Compute variational mutual information
        info_x_z += I_xz - exp(I_xz_perm - 1)
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
    # and latent prior distribution P(x)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + β * kl_qₓ_p - α * info_x_z / n_samples

end #function

@doc raw"""
    `infomax_loss(x, vae, mlp; σ, β, α, reconstruct, n_samples)`

Loss function for the infoMax variational autoencoder. The loss function is
defined as

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables. 

# Arguments
- `x::Vector`: Input to the neural network.
- `x_shuffle::Vector`: Shuffled input to the neural network needed to compute
  the mutual information term. This term is used to obtain an encoding
  `z_shuffle` that represents a random sample from the marginal P(z).
- `x_true::Vector`: True input against which to compare autoencoder
  reconstruction.
- `vae::VAE`: Struct containing the elements of the variational autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
  between input and output.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `α::Float32=1`: Annealing inverse temperature for the mutual information term.
- `n_samples::Int=1`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
  over multiple latent-space samples due to the variability of each sample. In
  practice, most algorithms keep this at 1.

# Returns
- `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
  when compared with reconstructed output `x̂`. This is used by the training
  algorithms to improve the reconstruction.
"""
function infomax_loss(
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32},
    x_true::AbstractVector{Float32},
    vae::VAE,
    mlp::Flux.Chain;
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
        # Run shuffled input through reconstruction function
        _, _, z_shuffle, _ = vae_reconstruct(
            x_shuffle, vae; latent=true
        )

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

        # Run input and latent variables through mutual information MLP
        I_xz = first(mlp([x; z]))
        # Run input and PERMUTED latent variables through mutual info MLP
        I_xz_perm = first(mlp([x; z_shuffle]))
        # Compute variational mutual information
        info_x_z += I_xz - exp(I_xz_perm - 1)
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
    # and latent prior distribution P(x)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + β * kl_qₓ_p - α * info_x_z / n_samples

end #function

@doc raw"""
    `infomlp_loss(x, vae, mlp; n_samples)`

Function used to train the multi-layered perceptron (mlp) used in the infoMaxVAE
algorithm to estimate the mutual information between the input x and the latent
space encoding z. The loss function is of the form

Ixz_MLP = g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩

The mutual information is expressed in a variational form (optimizing over the
space of all possible functions) where the MLP encodes the unknown optimal
function g(x, z).

# Arguments
- `x::Vector`: Input to the neural network.
- `x_shuffle::Vector`: Shuffled input to the neural network needed to compute
  the mutual information term. This term is used to obtain an encoding
  `z_shuffle` that represents a random sample from the marginal P(z).
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
  between input and output.

## Optional arguments
- `n_samples::Int=1`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
  over multiple latent-space samples due to the variability of each sample. In
  practice, most algorithms keep this at 1.

# Returns
- `Ixz_MLP::Float32`: Variational mutual information between input x and latent
  space encoding z.

"""
function infomlp_loss(
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32},
    vae::VAE,
    mlp::Flux.Chain;
    n_samples::Int=1
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
        # Run shuffled input through reconstruction function
        _, _, z_shuffle, _ = vae_reconstruct(
            x_shuffle, vae; latent=true
        )

        # Run input and latent variables through mutual information MLP
        I_xz = first(mlp([x; z]))
        # Run input and PERMUTED latent variables through mutual info MLP
        I_xz_perm = first(mlp([x; z_shuffle]))
        # Compute variational mutual information
        info_x_z += I_xz - exp(I_xz_perm - 1)
    end # for

    # Compute loss function
    return -info_x_z / n_samples

end #function

@doc raw"""
    `mutual_info_mlp(vae, mlp, data)`

Function to compute the mutual information between the input `x` and the latent
variable `z` for a given inforMaxVAE architecture.

# Arguments
- `vae::jlFitnotype.ml.VAE`: Struct containint the elements of a variational
    autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
    between input and output.
- `data::Matrix{Float32}`: Matrix containing the data on which to evaluate the
    loss function. NOTE: Every column should represent a single input.
"""
function mutual_info_mlp(
    vae::VAE, mlp::Flux.Chain, data::Matrix{Float32}
)
    # Generate list of random indexes for data shuffling
    shuffle_idx = Random.shuffle(1:size(data, 2))

    # Compute mutual information
    return StatsBase.mean(
        [-infomlp_loss(data[:, i], data[:, shuffle_idx[i]], vae, mlp)
         for i = 1:size(data, 2)]
    )

end # function

@doc raw"""
    `infomaxvae_train!(vae, mlp, data, opt; kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables.

# Arguments
- `vae::jlFitnotype.ml.VAE`: Struct containint the elements of a variational
  autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
  between input and output.
- `data::Matrix{Float32}`: Matrix containing the data on which to evaluate the
  loss function. NOTE: Every column should represent a single input.
- `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the variational autoencoder parameters. This should be fed already with
  the corresponding parameters. For example, one could feed: 
  ⋅ Flux.AMSGrad(η)
- `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the multi-layered perceptron parameters. This should be fed already
  with the corresponding parameters. For example, one could feed: 
  ⋅ Flux.AMSGrad(η)

## Optional arguments
- `kwargs::NamedTuple`: Tuple containing arguments for the loss function. For
    `infomax_loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.
    - `n_samples::Int=1`: Number of samples to take from the latent space when
      computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to
      average over multiple latent-space samples due to the variability of each
      sample. In practice, most algorithms keep this at 1.
"""
function infomaxvae_train!(
    vae::VAE,
    mlp::Flux.Chain,
    data::Matrix{Float32},
    vae_opt::Flux.Optimise.AbstractOptimiser,
    mlp_opt::Flux.Optimise.AbstractOptimiser;
    kwargs...
)
    # Extract VAE parameters
    vae_params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)
    # Extract MLP parameters
    mlp_params = Flux.params(mlp)

    # Generate list of random indexes for data shuffling
    shuffle_idx = Random.shuffle(1:size(data, 2))

    # Perform computation for first datum.
    # NOTE: This is to properly initialize the object on which the gradient will
    # be evaluated. There's probably better ways to do this, but this works.

    # == VAE == #
    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
        infomax_loss(data[:, 1], data[:, shuffle_idx[1]], vae, mlp; kwargs...)
    end # do

    # Having computed the pullback, we compute the loss function gradient
    ∇vae_loss_ = vae_back_(one(vae_loss_))

    # == MLP == #
    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
        infomlp_loss(data[:, 1], data[:, shuffle_idx[1]], vae, mlp)
    end # do

    # Having computed the pullback, we compute the loss function gradient
    ∇mlp_loss_ = mlp_back_(one(mlp_loss_))

    # Loop through the rest of the datasets data
    for (i, d) in enumerate(eachcol(data[:, 2:end]))
        # == VAE == #
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
            infomax_loss(d, data[:, shuffle_idx[i]], vae, mlp; kwargs...)
        end # do

        # Having computed the pullback, we compute the loss function gradient
        ∇vae_loss_ .+= vae_back_(one(vae_loss_))

        # == MLP == #
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
            infomlp_loss(d, data[:, shuffle_idx[i]], vae, mlp)
        end # do

        # Having computed the pullback, we compute the loss function gradient
        ∇mlp_loss_ .+= mlp_back_(one(mlp_loss_))
    end # for

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimise.update!(vae_opt, vae_params, ∇vae_loss_ ./ size(data, 2))

    # Update the MLP parameters averaging gradient from all datasets
    Flux.Optimise.update!(mlp_opt, mlp_params, ∇mlp_loss_ ./ size(data, 2))

end # function

@doc raw"""
    `infomaxvae_train!(vae, mlp, data, opt; kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables. For this method, the
data consists of a `Array{Float32, 3}` object, where the third dimension
contains both the noisy data and the "real" value against which to compare the
reconstruction.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
  between input and output.
- `data::Array{Float32, 3}`: Matrix containing the data on which to evaluate the
  loss function. NOTE: Every column should represent a single input.
- `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the variational autoencoder parameters. This should be fed already with
  the corresponding parameters. For example, one could feed: ⋅ Flux.AMSGrad(η)
- `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the multi-layered perceptron parameters. This should be fed already
  with the corresponding parameters. For example, one could feed: ⋅
  Flux.AMSGrad(η)

## Optional arguments
- `kwargs::NamedTuple`: Tuple containing arguments for the loss function. For
    `infomax_loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.
    - `n_samples::Int=1`: Number of samples to take from the latent space when
      computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to
      average over multiple latent-space samples due to the variability of each
      sample. In practice, most algorithms keep this at 1.
"""
function infomaxvae_train!(
    vae::VAE,
    mlp::Flux.Chain,
    data::Array{Float32,3},
    vae_opt::Flux.Optimise.AbstractOptimiser,
    mlp_opt::Flux.Optimise.AbstractOptimiser;
    kwargs...
)
    # Extract VAE parameters
    vae_params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)
    # Extract MLP parameters
    mlp_params = Flux.params(mlp)

    # Split data and real value
    data_noise = data[:, :, 1]
    data_true = data[:, :, 2]

    # Generate list of random indexes for data shuffling
    shuffle_idx = Random.shuffle(1:size(data, 2))

    # Perform computation for first datum.
    # NOTE: This is to properly initialize the object on which the gradient will
    # be evaluated. There's probably better ways to do this, but this works.

    # == VAE == #
    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
        infomax_loss(
            data_noise[:, 1],
            data_noise[:, shuffle_idx[1]],
            data_true[:, 1],
            vae,
            mlp;
            kwargs...
        )
    end # do

    # Having computed the pullback, we compute the loss function gradient
    ∇vae_loss_ = vae_back_(one(vae_loss_))

    # == MLP == #
    # Evaluate the loss function and compute the gradient. Zygote.pullback
    # gives two outputs: the result of the original function and a pullback,
    # which is the gradient of the function.
    mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
        infomlp_loss(data_noise[:, 1], data_noise[:, shuffle_idx[1]], vae, mlp)
    end # do

    # Having computed the pullback, we compute the loss function gradient
    ∇mlp_loss_ = mlp_back_(one(mlp_loss_))

    # Loop through the rest of the datasets data
    for i = 2:size(data_noise, 2)
        # == VAE == #
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
            infomax_loss(
                data_noise[:, i],
                data_noise[:, shuffle_idx[i]],
                data_true[:, i],
                vae,
                mlp;
                kwargs...
            )
        end # do

        # Having computed the pullback, we compute the loss function gradient
        ∇vae_loss_ .+= vae_back_(one(vae_loss_))

        # == MLP == #
        # Evaluate the loss function and compute the gradient. Zygote.pullback
        # gives two outputs: the result of the original function and a pullback,
        # which is the gradient of the function.
        mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
            infomlp_loss(
                data_noise[:, i], data_noise[:, shuffle_idx[i]], vae, mlp
            )
        end # do

        # Having computed the pullback, we compute the loss function gradient
        ∇mlp_loss_ .+= mlp_back_(one(mlp_loss_))
    end # for

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimise.update!(vae_opt, vae_params, ∇vae_loss_ ./ size(data, 2))

    # Update the MLP parameters averaging gradient from all datasets
    Flux.Optimise.update!(mlp_opt, mlp_params, ∇mlp_loss_ ./ size(data, 2))

end # function

@doc raw"""
    `step_scheduler(epoch, epoch_change, learning_rates)`

Simple function to define different learning rates at specified epochs.

# Arguments
- `epoch::Int`: Epoch at which to define learning rate.
- `epoch_change::Vector{<:Int}`: Number of epochs at which to change learning
  rate. It must include the initial learning rate!
- `learning_rates::Vector{<:AbstractFloat}`: Learning rate value for the epoch
  range. Must be the same length as `epoch_change`

# Returns
- `η::Abstr`
"""
function step_scheduler(
    epoch::Int,
    epoch_change::Vector{<:Int},
    learning_rates::Vector{<:AbstractFloat}
)
    # Check that arrays are of the same length
    if length(epoch_change) != length(learning_rates)
        error("epoch_change and learning_rates must be of the same length")
    end # if

    # Sort arrays to make sure it goes in epoch order.
    idx_sort = sortperm(epoch_change)
    sort!(epoch_change)
    learning_rates = learning_rates[idx_sort]

    # Check if there is any rate that belongs to the epoch
    if any(epoch .≤ epoch_change)
        # Return corresponding learning rate
        return first(learning_rates[epoch.≤epoch_change])
    else
        return learning_rates[end]
    end
end # function