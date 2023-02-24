# Setup
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


# Load packages
using LinearAlgebra
using Printf
using SparseArrays

using SpecialFunctions: erf

using SummationByPartsOperators
using Trixi
using OrdinaryDiffEq: CallbackSet

using PrettyTables: pretty_table, ft_printf
using FillArrays: Zeros
using LinearOperators: LinearOperator
using Krylov: Krylov

BLAS.set_num_threads(1)


# helper functions creating a (sparse) matrix representation from a matrix-free
# implementation in `f!`
function compute_linear_structure(f!, tmp, params, t)
  res = similar(tmp)

  tmp .= zero(eltype(tmp))
  f!(res, tmp, params, t)
  b = -vec(res)

  A = zeros(eltype(res), length(res), length(res))
  for j in 1:length(res)
    tmp[j] = one(eltype(tmp))
    f!(res, tmp, params, t)
    A[:, j] .= vec(res) .+ b
    tmp[j] = zero(eltype(tmp))
  end

  return A, b
end

function compute_sparse_linear_structure(f!, tmp, params, t)
  res = similar(tmp)
  vec_res = vec(res)

  tmp .= zero(eltype(tmp))
  f!(res, tmp, params, t)
  b = -vec(res)

  rowind = Vector{Int}()
  nzval = Vector{eltype(tmp)}()
  colptr = Vector{Int}(undef, length(tmp) + 1)

  for i in 1:length(res)
    tmp[i] = one(eltype(tmp))
    f!(res, tmp, params, t)
    @. vec_res = vec_res + b
    js = findall(!iszero, vec_res)
    colptr[i] = length(nzval) + 1
    if length(js) > 0
      append!(rowind, js)
      append!(nzval, vec_res[js])
    end
    tmp[i] = zero(eltype(tmp))
  end
  colptr[end] = length(nzval) + 1

  return SparseMatrixCSC(length(tmp), length(tmp), colptr, rowind, nzval), b
end


# 1D functions
function compute_q!(q, φ::AbstractArray{<:Real, 2}, params)
  (; basis, dl, dr, jacobian, Tᵣ, surface_values, φBC_left, φBC_right) = params
  sqrtTᵣ = sqrt(Tᵣ)

  # prolong φ, Dφ to surfaces
  for element in axes(φ, 2)
    left, right = element, element + 1
    inv_jacobian = inv(jacobian[element])
    surface_values[1, 2, left ] = φ[1, element]
    surface_values[2, 2, left ] = dot(dl, view(φ, :, element)) * inv_jacobian
    surface_values[1, 1, right] = φ[end, element]
    surface_values[2, 1, right] = dot(dr, view(φ, :, element)) * inv_jacobian
  end

  # compute q locally
  mul!(q, basis.D, φ)
  for element in axes(φ, 2)
    left, right = element, element + 1
    inv_jacobian = inv(jacobian[element])
    @. q[:, element] *= inv_jacobian

    # left boundary correction
    if element == 1
      if φBC_left === :periodic
        jump_φ  = surface_values[1, 2, left] - surface_values[1, 1, end]
        jump_Dφ = surface_values[2, 2, left] - surface_values[2, 1, end]
        inv_weight_r = inv(jacobian[element  ] * basis.weights[1])
        inv_weight_l = inv(jacobian[end      ] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q[1, element] += inv_jacobian / basis.weights[1] * (
            0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      else
        jump_φ  = surface_values[1, 2, left] - φBC_left
        q[1, element] += inv_jacobian / basis.weights[1] * jump_φ
      end
    else
      jump_φ  = surface_values[1, 2, left] - surface_values[1, 1, left]
      jump_Dφ = surface_values[2, 2, left] - surface_values[2, 1, left]
      inv_weight_r = inv(jacobian[element  ] * basis.weights[1])
      inv_weight_l = inv(jacobian[element-1] * basis.weights[end])
      denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
      c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
      c2 = inv(denominator)
      q[1, element] += inv_jacobian / basis.weights[1] * (
          0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
    end

    # right boundary correction
    if element == size(φ, 2)
      if φBC_right === :periodic
        jump_φ  = surface_values[1, 2, 1] - surface_values[1, 1, right]
        jump_Dφ = surface_values[2, 2, 1] - surface_values[2, 1, right]
        inv_weight_r = inv(jacobian[1        ] * basis.weights[1])
        inv_weight_l = inv(jacobian[element  ] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q[end, element] += inv_jacobian / basis.weights[end] * (
            0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      else
        jump_φ  = φBC_right - surface_values[1, 1, right]
        q[end, element] += inv_jacobian / basis.weights[end] * jump_φ
      end
    else
      jump_φ  = surface_values[1, 2, right] - surface_values[1, 1, right]
      jump_Dφ = surface_values[2, 2, right] - surface_values[2, 1, right]
      inv_weight_r = inv(jacobian[element+1] * basis.weights[1])
      inv_weight_l = inv(jacobian[element  ] * basis.weights[end])
      denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
      c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
      c2 = inv(denominator)
      q[end, element] += inv_jacobian / basis.weights[end] * (
          0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
    end
  end

  return nothing
end

function compute_div_q!(div_q, φ::AbstractArray{<:Real, 2}, q, params)
  (; basis, jacobian, Tᵣ, surface_values, φBC_left, φBC_right) = params
  sqrtTᵣ = sqrt(Tᵣ)
  inv_sqrtTᵣ = inv(sqrtTᵣ)

  # prolong φ, q to surfaces
  for element in axes(φ, 2)
    left, right = element, element+1
    surface_values[1, 2, left ] = φ[1, element]
    surface_values[2, 2, left ] = q[1, element]
    surface_values[1, 1, right] = φ[end, element]
    surface_values[2, 1, right] = q[end, element]
  end

  # compute div q locally
  mul!(div_q, basis.D, q)
  for element in axes(φ, 2)
    left, right = element, element+1
    inv_jacobian = inv(jacobian[element])
    @. div_q[:, element] *= inv_jacobian

    # left boundary correction
    if element == 1
      if φBC_left === :periodic
        jump_φ = surface_values[1, 2, left] - surface_values[1, 1, end]
        jump_q = surface_values[2, 2, left] - surface_values[2, 1, end]
        div_q[1, element] -= inv_jacobian / basis.weights[1] * (
            -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      else
        jump_φ = surface_values[1, 2, left] - φBC_left
        div_q[1, element] += inv_jacobian / basis.weights[1] * 0.5 * inv_sqrtTᵣ * jump_φ
      end
    else
      jump_φ = surface_values[1, 2, left] - surface_values[1, 1, left]
      jump_q = surface_values[2, 2, left] - surface_values[2, 1, left]
      div_q[1, element] -= inv_jacobian / basis.weights[1] * (
          -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
    end

    # right boundary correction
    if element == size(φ, 2)
      if φBC_right === :periodic
        jump_φ = surface_values[1, 2, 1] - surface_values[1, 1, right]
        jump_q = surface_values[2, 2, 1] - surface_values[2, 1, right]
        div_q[end, element] += inv_jacobian / basis.weights[end] * (
            0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      else
        jump_φ = φBC_right - surface_values[1, 1, right]
        div_q[end, element] -= inv_jacobian / basis.weights[end] * 0.5 * inv_sqrtTᵣ * jump_φ
      end
    else
      jump_φ = surface_values[1, 2, right] - surface_values[1, 1, right]
      jump_q = surface_values[2, 2, right] - surface_values[2, 1, right]
      div_q[end, element] += inv_jacobian / basis.weights[end] * (
          0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
    end
  end

  return nothing
end


# 2D functions
function compute_q!(q, φ::AbstractArray{<:Real, 4}, params)
  (; basis, dl, dr, Tᵣ,
     jacobian_x, jacobian_y,
     surface_values_x, surface_values_y,
     φBC_left, φBC_right, φBC_bottom, φBC_top) = params
  sqrtTᵣ = sqrt(Tᵣ)
  q1, q2 = q

  # prolong φ, Dφ to surfaces
  for element_y in axes(φ, 4), element_x in axes(φ, 3)
    left, right = element_x, element_x + 1
    bottom, top = element_y, element_y + 1
    inv_jacobian_x = inv(jacobian_x[element_x])
    inv_jacobian_y = inv(jacobian_y[element_y])

    # x direction
    for j in axes(φ, 2)
      surface_values_x[1, j, 2, left,   element_y] = φ[1,   j, element_x, element_y]
      surface_values_x[2, j, 2, left,   element_y] = dot(dl, view(φ, :, j, element_x, element_y)) * inv_jacobian_x
      surface_values_x[1, j, 1, right,  element_y] = φ[end, j, element_x, element_y]
      surface_values_x[2, j, 1, right,  element_y] = dot(dr, view(φ, :, j, element_x, element_y)) * inv_jacobian_x
    end

    # y direction
    for i in axes(φ, 1)
      surface_values_y[1, i, 2, bottom, element_x] = φ[i, 1,   element_x, element_y]
      surface_values_y[2, i, 2, bottom, element_x] = dot(dl, view(φ, i, :, element_x, element_y)) * inv_jacobian_y
      surface_values_y[1, i, 1, top,    element_x] = φ[i, end, element_x, element_y]
      surface_values_y[2, i, 1, top,    element_x] = dot(dr, view(φ, i, :, element_x, element_y)) * inv_jacobian_y
    end
  end

  # compute q1 locally
  for element_y in axes(φ, 4), element_x in axes(φ, 3)
    left, right = element_x, element_x + 1
    inv_jacobian = inv(jacobian_x[element_x])

    for j in axes(φ, 2)
      mul!(view(q1, :, j, element_x, element_y), basis.D,
           view(φ,  :, j, element_x, element_y), inv_jacobian, false)

      # left boundary correction
      if element_x == 1
        if φBC_left === :periodic
          jump_φ  = surface_values_x[1, j, 2, left, element_y] - surface_values_x[1, j, 1, end, element_y]
          jump_Dφ = surface_values_x[2, j, 2, left, element_y] - surface_values_x[2, j, 1, end, element_y]
          inv_weight_r = inv(jacobian_x[element_x] * basis.weights[1])
          inv_weight_l = inv(jacobian_x[end      ] * basis.weights[end])
          denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
          c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
          c2 = inv(denominator)
          q1[1, j, element_x, element_y] += inv_jacobian / basis.weights[1] * (
              0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
        else
          jump_φ = surface_values_x[1, j, 2, left, element_y] - φBC_left[j, element_y]
          q1[1, j, element_x, element_y] += inv_jacobian / basis.weights[1] * jump_φ
        end
      else
        jump_φ  = surface_values_x[1, j, 2, left, element_y] - surface_values_x[1, j, 1, left, element_y]
        jump_Dφ = surface_values_x[2, j, 2, left, element_y] - surface_values_x[2, j, 1, left, element_y]
        inv_weight_r = inv(jacobian_x[element_x  ] * basis.weights[1])
        inv_weight_l = inv(jacobian_x[element_x-1] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q1[1, j, element_x, element_y] += inv_jacobian / basis.weights[1] * (
            0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      end

      # right boundary correction
      if element_x == size(φ, 3)
        if φBC_right === :periodic
          jump_φ  = surface_values_x[1, j, 2, 1, element_y] - surface_values_x[1, j, 1, right, element_y]
          jump_Dφ = surface_values_x[2, j, 2, 1, element_y] - surface_values_x[2, j, 1, right, element_y]
          inv_weight_r = inv(jacobian_x[1        ] * basis.weights[1])
          inv_weight_l = inv(jacobian_x[element_x] * basis.weights[end])
          denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
          c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
          c2 = inv(denominator)
          q1[end, j, element_x, element_y] += inv_jacobian / basis.weights[end] * (
              0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
        else
          jump_φ = φBC_right[j, element_y] - surface_values_x[1, j, 1, right, element_y]
          q1[end, j, element_x, element_y] += inv_jacobian / basis.weights[end] * jump_φ
        end
      else
        jump_φ  = surface_values_x[1, j, 2, right, element_y] - surface_values_x[1, j, 1, right, element_y]
        jump_Dφ = surface_values_x[2, j, 2, right, element_y] - surface_values_x[2, j, 1, right, element_y]
        inv_weight_r = inv(jacobian_x[element_x+1] * basis.weights[1])
        inv_weight_l = inv(jacobian_x[element_x  ] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q1[end, j, element_x, element_y] += inv_jacobian / basis.weights[end] * (
            0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      end
    end
  end

  # compute q2 locally
  for element_y in axes(φ, 4), element_x in axes(φ, 3)
    bottom, top = element_y, element_y + 1
    inv_jacobian = inv(jacobian_y[element_y])

    for i in axes(φ, 1)
      mul!(view(q2, i, :, element_x, element_y), basis.D,
           view(φ,  i, :, element_x, element_y), inv_jacobian, false)

      # bottom boundary correction
      if element_y == 1
        if φBC_bottom === :periodic
          jump_φ  = surface_values_y[1, i, 2, bottom, element_x] - surface_values_y[1, i, 1, end, element_x]
          jump_Dφ = surface_values_y[2, i, 2, bottom, element_x] - surface_values_y[2, i, 1, end, element_x]
          inv_weight_r = inv(jacobian_y[element_y] * basis.weights[1])
          inv_weight_l = inv(jacobian_y[end      ] * basis.weights[end])
          denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
          c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
          c2 = inv(denominator)
          q2[i, 1, element_x, element_y] += inv_jacobian / basis.weights[1] * (
              0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
        else
          jump_φ = surface_values_y[1, i, 2, bottom, element_x] - φBC_bottom[i, element_x]
          q2[i, 1, element_x, element_y] += inv_jacobian / basis.weights[1] * jump_φ
        end
      else
        jump_φ  = surface_values_y[1, i, 2, bottom, element_x] - surface_values_y[1, i, 1, bottom, element_x]
        jump_Dφ = surface_values_y[2, i, 2, bottom, element_x] - surface_values_y[2, i, 1, bottom, element_x]
        inv_weight_r = inv(jacobian_y[element_y  ] * basis.weights[1])
        inv_weight_l = inv(jacobian_y[element_y-1] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q2[i, 1, element_x, element_y] += inv_jacobian / basis.weights[1] * (
            0.5 * (1 + sqrtTᵣ * c1) * jump_φ - 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      end

      # top boundary correction
      if element_y == size(φ, 4)
        if φBC_top === :periodic
          jump_φ  = surface_values_y[1, i, 2, 1, element_x] - surface_values_y[1, i, 1, top, element_x]
          jump_Dφ = surface_values_y[2, i, 2, 1, element_x] - surface_values_y[2, i, 1, top, element_x]
          inv_weight_r = inv(jacobian_y[1        ] * basis.weights[1])
          inv_weight_l = inv(jacobian_y[element_y] * basis.weights[end])
          denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
          c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
          c2 = inv(denominator)
          q2[i, end, element_x, element_y] += inv_jacobian / basis.weights[end] * (
              0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
        else
          jump_φ = φBC_top[i, element_x] - surface_values_y[1, i, 1, top, element_x]
          q2[i, end, element_x, element_y] += inv_jacobian / basis.weights[end] * jump_φ
        end
      else
        jump_φ  = surface_values_y[1, i, 2, top, element_x] - surface_values_y[1, i, 1, top, element_x]
        jump_Dφ = surface_values_y[2, i, 2, top, element_x] - surface_values_y[2, i, 1, top, element_x]
        inv_weight_r = inv(jacobian_y[element_y+1] * basis.weights[1])
        inv_weight_l = inv(jacobian_y[element_y  ] * basis.weights[end])
        denominator = 1 + 0.5 * sqrtTᵣ * (inv_weight_r + inv_weight_l)
        c1 = 0.5 * (inv_weight_r - inv_weight_l) / denominator
        c2 = inv(denominator)
        q2[i, end, element_x, element_y] += inv_jacobian / basis.weights[end] * (
            0.5 * (1 - sqrtTᵣ * c1) * jump_φ + 0.5 * sqrtTᵣ * c2 * jump_Dφ)
      end
    end
  end

  return nothing
end

function compute_div_q!(div_q, φ::AbstractArray{<:Real, 4}, q, params)
  (; basis, Tᵣ,
     jacobian_x, jacobian_y,
     surface_values_x, surface_values_y,
     φBC_left, φBC_right, φBC_bottom, φBC_top) = params
  sqrtTᵣ = sqrt(Tᵣ)
  inv_sqrtTᵣ = inv(sqrtTᵣ)
  q1, q2 = q

  # prolong φ, q to surfaces
  for element_y in axes(φ, 4), element_x in axes(φ, 3)
    left, right = element_x, element_x + 1
    bottom, top = element_y, element_y + 1

    # x direction
    for j in axes(φ, 2)
      surface_values_x[1, j, 2, left ,  element_y] = φ[1,    j, element_x, element_y]
      surface_values_x[2, j, 2, left ,  element_y] = q1[1,   j, element_x, element_y]
      surface_values_x[1, j, 1, right,  element_y] = φ[end,  j, element_x, element_y]
      surface_values_x[2, j, 1, right,  element_y] = q1[end, j, element_x, element_y]
    end

    # y direction
    for i in axes(φ, 1)
      surface_values_y[1, i, 2, bottom, element_x] = φ[i,  1,   element_x, element_y]
      surface_values_y[2, i, 2, bottom, element_x] = q2[i, 1,   element_x, element_y]
      surface_values_y[1, i, 1, top   , element_x] = φ[i,  end, element_x, element_y]
      surface_values_y[2, i, 1, top   , element_x] = q2[i, end, element_x, element_y]
    end
  end

  # compute div q locally
  for element_y in axes(φ, 4), element_x in axes(φ, 3)
    left, right = element_x, element_x + 1
    bottom, top = element_y, element_y + 1
    inv_jacobian_x = inv(jacobian_x[element_x])
    inv_jacobian_y = inv(jacobian_y[element_y])

    # x direction
    for j in axes(φ, 2)
      mul!(view(div_q, :, j, element_x, element_y), basis.D,
           view(q1,    :, j, element_x, element_y), inv_jacobian_x, false)

      # left boundary correction
      if element_x == 1
        if φBC_left === :periodic
          jump_φ = surface_values_x[1, j, 2, left, element_y] - surface_values_x[1, j, 1, end, element_y]
          jump_q = surface_values_x[2, j, 2, left, element_y] - surface_values_x[2, j, 1, end, element_y]
          div_q[1, j, element_x, element_y] -= inv_jacobian_x / basis.weights[1] * (
              -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
        else
          jump_φ = surface_values_x[1, j, 2, left, element_y] - φBC_left[j, element_y]
          div_q[1, j, element_x, element_y] += inv_jacobian_x / basis.weights[1] * 0.5 * inv_sqrtTᵣ * jump_φ
        end
      else
        jump_φ = surface_values_x[1, j, 2, left, element_y] - surface_values_x[1, j, 1, left, element_y]
        jump_q = surface_values_x[2, j, 2, left, element_y] - surface_values_x[2, j, 1, left, element_y]
        div_q[1, j, element_x, element_y] -= inv_jacobian_x / basis.weights[1] * (
            -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      end

      # right boundary correction
      if element_x == size(φ, 3)
        if φBC_right === :periodic
          jump_φ = surface_values_x[1, j, 2, 1, element_y] - surface_values_x[1, j, 1, right, element_y]
          jump_q = surface_values_x[2, j, 2, 1, element_y] - surface_values_x[2, j, 1, right, element_y]
          div_q[end, j, element_x, element_y] += inv_jacobian_x / basis.weights[end] * (
              0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
        else
          jump_φ = φBC_right[j, element_y] - surface_values_x[1, j, 1, right, element_y]
          div_q[end, j, element_x, element_y] -= inv_jacobian_x / basis.weights[end] * 0.5 * inv_sqrtTᵣ * jump_φ
        end
      else
        jump_φ = surface_values_x[1, j, 2, right, element_y] - surface_values_x[1, j, 1, right, element_y]
        jump_q = surface_values_x[2, j, 2, right, element_y] - surface_values_x[2, j, 1, right, element_y]
        div_q[end, j, element_x, element_y] += inv_jacobian_x / basis.weights[end] * (
            0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      end
    end

    # y direction
    for i in axes(φ, 1)
      mul!(view(div_q, i, :, element_x, element_y), basis.D,
           view(q2,    i, :, element_x, element_y), inv_jacobian_y, true)

      # bottom boundary correction
      if element_y == 1
        if φBC_bottom === :periodic
          jump_φ = surface_values_y[1, i, 2, bottom, element_x] - surface_values_y[1, i, 1, end, element_x]
          jump_q = surface_values_y[2, i, 2, bottom, element_x] - surface_values_y[2, i, 1, end, element_x]
          div_q[i, 1, element_x, element_y] -= inv_jacobian_y / basis.weights[1] * (
              -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
        else
          jump_φ = surface_values_y[1, i, 2, bottom, element_x] - φBC_bottom[i, element_x]
          div_q[i, 1, element_x, element_y] += inv_jacobian_y / basis.weights[1] * 0.5 * inv_sqrtTᵣ * jump_φ
        end
      else
        jump_φ = surface_values_y[1, i, 2, bottom, element_x] - surface_values_y[1, i, 1, bottom, element_x]
        jump_q = surface_values_y[2, i, 2, bottom, element_x] - surface_values_y[2, i, 1, bottom, element_x]
        div_q[i, 1, element_x, element_y] -= inv_jacobian_y / basis.weights[1] * (
            -0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      end

      # top boundary correction
      if element_y == size(φ, 4)
        if φBC_top === :periodic
          jump_φ = surface_values_y[1, i, 2, 1, element_x] - surface_values_y[1, i, 1, top, element_x]
          jump_q = surface_values_y[2, i, 2, 1, element_x] - surface_values_y[2, i, 1, top, element_x]
          div_q[i, end, element_x, element_y] += inv_jacobian_y / basis.weights[end] * (
              0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
        else
          jump_φ = φBC_top[i, element_x] - surface_values_y[1, i, 1, top, element_x]
          div_q[i, end, element_x, element_y] -= inv_jacobian_y / basis.weights[end] * 0.5 * inv_sqrtTᵣ * jump_φ
        end
      else
        jump_φ = surface_values_y[1, i, 2, top, element_x] - surface_values_y[1, i, 1, top, element_x]
        jump_q = surface_values_y[2, i, 2, top, element_x] - surface_values_y[2, i, 1, top, element_x]
        div_q[i, end, element_x, element_y] += inv_jacobian_y / basis.weights[end] * (
            0.5 * jump_q + 0.5 * inv_sqrtTᵣ * jump_φ)
      end
    end
  end

  return nothing
end


function hypdiff_parabolic!(dφ, φ, params, t)
  (; f, q) = params

  compute_q!(q, φ, params)

  compute_div_q!(dφ, φ, q, params)

  @. dφ = dφ + f

  return nothing
end

function hypdiff_parabolic_for_cg!(dφ_vector, φ_vector, params)
  (; q) = params
  if q isa Array
    # 1D
    s = size(q)
  else
    # 2D
    s = size(q[1])
  end
  dφ = reshape(dφ_vector, s)
  φ = reshape(φ_vector, s)

  # need to set Dirichlet boundary conditions to homogeneous since the
  # non-homogeneous BCs are contained in the right-hand side vector
  params_homogeneous = params
  if haskey(params, :φBC_left)
    if params.φBC_left isa Number
      params_homogeneous = (; params_homogeneous..., φBC_left = zero(params.φBC_left))
    elseif params.φBC_left isa AbstractArray
      params_homogeneous = (; params_homogeneous..., φBC_left = Zeros(size(params.φBC_left)))
    end
  end
  if haskey(params, :φBC_right)
    if params.φBC_right isa Number
      params_homogeneous = (; params_homogeneous..., φBC_right = zero(params.φBC_right))
    elseif params.φBC_right isa AbstractArray
      params_homogeneous = (; params_homogeneous..., φBC_right = Zeros(size(params.φBC_right)))
    end
  end
  if haskey(params, :φBC_bottom)
    if params.φBC_bottom isa Number
      params_homogeneous = (; params_homogeneous..., φBC_bottom = zero(params.φBC_bottom))
    elseif params.φBC_bottom isa AbstractArray
      params_homogeneous = (; params_homogeneous..., φBC_bottom = Zeros(size(params.φBC_bottom)))
    end
  end
  if haskey(params, :φBC_top)
    if params.φBC_top isa Number
      params_homogeneous = (; params_homogeneous..., φBC_top = zero(params.φBC_top))
    elseif params.φBC_top isa AbstractArray
      params_homogeneous = (; params_homogeneous..., φBC_top = Zeros(size(params.φBC_top)))
    end
  end

  compute_q!(q, φ, params_homogeneous)

  compute_div_q!(dφ, φ, q, params_homogeneous)

  mul_by_mass_matrix!(dφ, params_homogeneous)

  return nothing
end


# 1D function
function mul_by_mass_matrix!(u::AbstractArray{<:Real, 2}, params)
  (; basis, jacobian) = params
  weights = basis.weights

  for element in axes(u, 2)
    factor = jacobian[element]
    for i in axes(u, 1)
      u[i, element] *= factor * weights[i]
    end
  end

  return nothing
end

# 2D function
function mul_by_mass_matrix!(u::AbstractArray{<:Real, 4}, params)
  (; basis, jacobian_x, jacobian_y) = params
  weights = basis.weights

  for element_y in axes(u, 4), element_x in axes(u, 3)
    factor = jacobian_x[element_x] * jacobian_y[element_y]
    for j in axes(u, 2), i in axes(u, 1)
      u[i, j, element_x, element_y] *= factor * weights[i] * weights[j]
    end
  end

  return nothing
end


# 1D functions
function l2_norm(u1, u2::AbstractArray{<:Real, 2}, params)
  (; basis, jacobian) = params
  (; weights) = basis

  res = zero((u1[1] - u2[1]) * weights[1])
  for element in axes(u1, 2)
    factor = jacobian[element]
    for i in axes(u1, 1)
      res += factor * weights[i] * (u1[i, element] - u2[i, element])^2
    end
  end

  return sqrt(res)
end

function integral(u::AbstractArray{<:Real, 2}, params)
  (; basis, jacobian) = params
  (; weights) = basis

  res = zero(u[1] * weights[1])
  for element in axes(u, 2)
    factor = jacobian[element]
    for i in axes(u, 1)
      res += factor * weights[i] * u[i, element]
    end
  end

  return res
end


# 2D functions
function l2_norm(u1, u2::AbstractArray{<:Real, 4}, params)
  (; basis, jacobian_x, jacobian_y) = params
  (; weights) = basis

  res = zero((u1[1] - u2[1]) * weights[1])
  for element_y in axes(u1, 4), element_x in axes(u1, 3)
    factor = jacobian_x[element_x] * jacobian_y[element_y]
    for j in axes(u1, 2), i in axes(u1, 1)
      res += factor * weights[i] * weights[j] *
                (u1[i, j, element_x, element_y] - u2[i, j, element_x, element_y])^2
    end
  end

  return sqrt(res)
end

function integral(u::AbstractArray{<:Real, 4}, params)
  (; basis, jacobian_x, jacobian_y) = params
  (; weights) = basis

  res = zero(u[1] * weights[1])
  for element_y in axes(u, 4), element_x in axes(u, 3)
    factor = jacobian_x[element_x] * jacobian_y[element_y]
    for j in axes(u, 2), i in axes(u, 1)
      res += factor * weights[i] * weights[j] * u[i, j, element_x, element_y]
    end
  end

  return res
end


# utility function
function compute_eoc(Ns, errors)
  eoc = similar(errors)
  eoc[begin] = NaN # no EOC defined for the first grid
  for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
    eoc[idx] = -( log(errors[idx] / errors[idx - 1]) / log(Ns[idx] / Ns[idx - 1]) )
  end
  return eoc
end


# 1D functions
function setup_1d(xmin, xmax, polydeg, N)
  Lᵣ = (xmax - xmin) / (2π)
  Tᵣ = Lᵣ^2 # std. choice of Nishikawa, this makes all modes underdamped
  # Tᵣ = Lᵣ^2 / 4 # this makes the lowest mode damped ideally

  basis = LobattoLegendre(polydeg)
  element_boundaries = range(xmin, stop=xmax, length=N+1) # uniform mesh

  x = zeros(length(basis.weights), length(element_boundaries) - 1)
  jacobian = zeros(length(element_boundaries) - 1)
  for element in axes(x, 2)
    jac = (element_boundaries[element+1] - element_boundaries[element]) /
          (basis.nodes[end] - basis.nodes[1])
    @. x[:, element] = element_boundaries[element] + (basis.nodes - basis.nodes[1]) * jac
    jacobian[element] = jac
  end

  surface_values = similar(x, 2, 2, length(element_boundaries)) # (φ,q), (left,right), surface

  dl = basis.D[1, :]
  dr = basis.D[end, :]

  return (; basis, dl, dr, jacobian, x, Tᵣ, surface_values)
end

function solve_1d_nonperiodic(; polydeg = 3, N = 10)
  xmin, xmax = -1.0, 1.0

  φ_func(x) = exp(-10 * x^2)
  q_func(x) = -20 * x * exp(-10 * x^2)
  f_func(x) = -20 * (20 * x^2 - 1) * exp(-10 * x^2)

  (; basis, dl, dr, jacobian, x, Tᵣ, surface_values) = setup_1d(
    xmin, xmax, polydeg, N)

  φ0 = zero(x)
  q  = similar(φ0)
  f  = f_func.(x)
  tmp = similar(φ0)
  φsol = φ_func.(x)
  qsol = q_func.(x)
  φBC_left  = φ_func(first(x))
  φBC_right = φ_func(last(x))
  params = (; basis, dl, dr, jacobian, x, Tᵣ, surface_values, f, q, φBC_left, φBC_right, tmp, φsol, qsol)

  A, b = compute_sparse_linear_structure(hypdiff_parabolic!, copy(φ0), params, 0.0)
  φ = reshape(A \ b, size(φ0))
  q = similar(φ)
  compute_q!(q, φ, params)

  error_φ = l2_norm(φ, φsol, params)
  error_q = l2_norm(q, qsol, params)
  return (; error_φ, error_q, x, φ, q, φsol, qsol, params)
end

function solve_1d_periodic(; polydeg = 3, N = 10)
  xmin, xmax = -2.0, 2.0

  # needs to have a vanishing mean value for periodic BCs - assume the domain
  # is symmetric around the origin
  φ_func(x) = exp(-10 * x^2) - sqrt(pi / 10) * erf(sqrt(10) * xmax) / (xmax - xmin)
  q_func(x) = -20 * x * exp(-10 * x^2)
  f_func(x) = -20 * (20 * x^2 - 1) * exp(-10 * x^2)

  (; basis, dl, dr, jacobian, x, Tᵣ, surface_values) = setup_1d(
    xmin, xmax, polydeg, N)

  φ0 = zero(x)
  q  = similar(φ0)
  f  = f_func.(x)
  tmp = similar(φ0)
  φsol = φ_func.(x)
  qsol = q_func.(x)
  φBC_left  = :periodic
  φBC_right = :periodic
  params = (; basis, dl, dr, jacobian, x, Tᵣ, surface_values, f, q, φBC_left, φBC_right, tmp, φsol, qsol)

  A, b = compute_sparse_linear_structure(hypdiff_parabolic!, copy(φ0), params, 0.0)
  φ = reshape(A \ b, size(φ0))
  # subtract mean value since direct solvers may not give the solution with
  # vanishing mean value but, e.g., the least norm solution
  φ .= φ .- integral(φ, params) / (xmax - xmin)
  q = similar(φ)
  compute_q!(q, φ, params)

  error_φ = l2_norm(φ, φsol, params)
  error_q = l2_norm(q, qsol, params)
  return (; error_φ, error_q, x, φ, q, φsol, qsol, params)
end


@inline function Trixi.flux_godunov(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations1D)
  # Obtain left and right fluxes
  phi_ll, q1_ll = u_ll
  phi_rr, q1_rr = u_rr
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (q1_rr - q1_ll)

  return SVector(f1, f2)
end

function solve_1d_nonperiodic_trixi(; polydeg = 3, N = 10,
                                      surface_flux = flux_godunov)
  xmin, xmax = -1.0, 1.0

  φ_func(x) = exp(-10 * x^2)
  q_func(x) = -20 * x * exp(-10 * x^2)
  f_func(x) = -20 * (20 * x^2 - 1) * exp(-10 * x^2)

  initial_condition(x, t, equations) = if iszero(t)
    return SVector(zero(t), zero(t))
  else
    return SVector(φ_func(x[1]), q_func(x[1]))
  end
  source_terms(u, x, t, equations) = let
    harmonic = source_terms_harmonic(u, x, t, equations)
    return SVector(f_func(x[1]), 0) + harmonic
  end
  boundary_condition_nonperiodic(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations) = let
    u_boundary = initial_condition(x, one(t), equations)
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right"
      flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    return flux
  end

  equations = HyperbolicDiffusionEquations1D(Lr = (xmax - xmin) / (2π))
  solver = DGSEM(; polydeg, surface_flux)
  mesh = StructuredMesh((N,), (xmin,), (xmax,), periodicity=(false,))
  semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition, solver;
    source_terms, boundary_conditions = boundary_condition_nonperiodic)

  ode = semidiscretize(semi, (0.0, 100.0))
  steady_state_callback = SteadyStateCallback(abstol = 5.0e-12, reltol = 0.0)
  stepsize_callback = StepsizeCallback(cfl = 1.5)
  callbacks = CallbackSet(steady_state_callback, stepsize_callback)
  sol = redirect_stderr(devnull) do
    # avoid cluttering the REPL with output of the steady_state_callback
    Trixi.solve(
      ode, Trixi.HypDiffN3Erk3Sstar52(),
      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
      save_everystep = false, callback = callbacks)
  end

  usol = Trixi.compute_coefficients(initial_condition, 1.0, semi)
  diff = sol.u[end] - usol
  error_φ = integrate(abs2 ∘ ((u, equations) -> u[1]), diff, semi; normalize = false) |> sqrt
  error_q = integrate(abs2 ∘ ((u, equations) -> u[2]), diff, semi; normalize = false) |> sqrt
  return (; error_φ, error_q, sol, usol, semi)
end

function solve_1d_periodic_trixi(; polydeg = 3, N = 10,
                                   surface_flux = flux_godunov)
  xmin, xmax = -2.0, 2.0

  # needs to have a vanishing mean value for periodic BCs - assume the domain
  # is symmetric around the origin
  φ_func(x) = exp(-10 * x^2) - sqrt(pi / 10) * erf(sqrt(10) * xmax) / (xmax - xmin)
  q_func(x) = -20 * x * exp(-10 * x^2)
  f_func_tmp(x) = -20 * (20 * x^2 - 1) * exp(-10 * x^2)

  initial_condition(x, t, equations) = if iszero(t)
    return SVector(zero(t), zero(t))
  else
    return SVector(φ_func(x[1]), q_func(x[1]))
  end

  equations = HyperbolicDiffusionEquations1D(Lr = (xmax - xmin) / (2π))
  # equations = HyperbolicDiffusionEquations1D(Lr = 1.0)
  solver = DGSEM(; polydeg, surface_flux)
  mesh = StructuredMesh((N,), (xmin,), (xmax,), periodicity=(true,))
  semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition, solver)

  # the source term must have discretely vanishing mean value for periodic BCs
  source = Trixi.compute_coefficients(1.0, semi) do x, t, equations
    SVector(f_func_tmp(x[1]), 0)
  end
  mean_value = integrate(first, source, semi)
  f_func(x) = let mean_value = mean_value
    return -20 * (20 * x^2 - 1) * exp(-10 * x^2) - mean_value
  end

  source_terms(u, x, t, equations) = let
    harmonic = source_terms_harmonic(u, x, t, equations)
    return SVector(f_func(x[1]), 0) + harmonic
  end

  semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition, solver;
    source_terms, boundary_conditions = boundary_condition_periodic)

  ode = semidiscretize(semi, (0.0, 1.0e3))
  steady_state_callback = SteadyStateCallback(abstol = 5.0e-12, reltol = 0.0)
  stepsize_callback = StepsizeCallback(cfl = 1.5)
  callbacks = CallbackSet(steady_state_callback, stepsize_callback)
  sol = redirect_stderr(devnull) do
    # avoid cluttering the REPL with output of the steady_state_callback
    Trixi.solve(
      ode, Trixi.HypDiffN3Erk3Sstar52(),
      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
      save_everystep = false, callback = callbacks)
  end

  usol = Trixi.compute_coefficients(initial_condition, 1.0, semi)
  diff = sol.u[end] - usol
  error_φ = integrate(abs2 ∘ ((u, equations) -> u[1]), diff, semi; normalize = false) |> sqrt
  error_q = integrate(abs2 ∘ ((u, equations) -> u[2]), diff, semi; normalize = false) |> sqrt
  return (; error_φ, error_q, sol, usol, semi, source)
end


# 2D functions
function setup_2d(xmin, xmax, ymin, ymax, polydeg, Nx, Ny)
  # equation (26) of Nishikawa and Nakashima (2018)
  Lᵣ = (xmax - xmin) * (ymax - ymin) / (2π * sqrt((xmax - xmin)^2 + (ymax - ymin)^2))
  Tᵣ = Lᵣ^2 # std. choice of Nishikawa, this makes all modes underdamped
  # Tᵣ = Lᵣ^2 / 4 # this makes the lowest mode damped ideally

  basis = LobattoLegendre(polydeg)
  element_boundaries_x = range(xmin, stop = xmax, length = Nx + 1) # uniform mesh
  element_boundaries_y = range(ymin, stop = ymax, length = Ny + 1) # uniform mesh

  x = zeros(length(basis.weights), length(basis.weights),
            length(element_boundaries_x) - 1, length(element_boundaries_y) - 1)
  y = zeros(length(basis.weights), length(basis.weights),
            length(element_boundaries_x) - 1, length(element_boundaries_y) - 1)
  jacobian_x = zeros(length(element_boundaries_x) - 1)
  jacobian_y = zeros(length(element_boundaries_y) - 1)
  for element_x in axes(x, 3)
    jac = (element_boundaries_x[element_x+1] - element_boundaries_x[element_x]) /
          (basis.nodes[end] - basis.nodes[1])
    jacobian_x[element_x] = jac
    for element_y in axes(x, 4), j in axes(x, 2)
      @. x[:, j, element_x, element_y] = element_boundaries_x[element_x] + (basis.nodes - basis.nodes[1]) * jac
    end
  end
  for element_y in axes(y, 4)
    jac = (element_boundaries_y[element_y+1] - element_boundaries_y[element_y]) /
          (basis.nodes[end] - basis.nodes[1])
    jacobian_y[element_y] = jac
    for element_x in axes(y, 3), i in axes(y, 1)
      @. y[i, :, element_x, element_y] = element_boundaries_y[element_y] + (basis.nodes - basis.nodes[1]) * jac
    end
  end

  # (φ,q), other_nodes, (left,right), surface, eleemnt_in_other_direction
  surface_values_x = similar(x, 2, size(x, 2), 2, length(element_boundaries_x), size(x, 4))
  surface_values_y = similar(x, 2, size(x, 1), 2, length(element_boundaries_y), size(x, 3))

  dl = basis.D[1, :]
  dr = basis.D[end, :]

  return (; basis, dl, dr, Tᵣ,
            jacobian_x, jacobian_y,
            x, y,
            surface_values_x, surface_values_y)
end

function solve_2d_nonperiodic(; polydeg = 3, N = 10, Nx = N, Ny = N)
  # setup of Cockburn, Guzman, Wang (2008)
  xmin, xmax = -0.5, 0.5
  ymin, ymax = -0.5, 0.5
  φ_func(x, y)  =           cospi(x) * cospi(y)
  q1_func(x, y) =    -π   * sinpi(x) * cospi(y)
  q2_func(x, y) =    -π   * cospi(x) * sinpi(y)
  f_func(x, y)  = 2 * π^2 * cospi(x) * cospi(y)

  (; basis, dl, dr, Tᵣ, jacobian_x, jacobian_y, x, y,
     surface_values_x, surface_values_y) = setup_2d(xmin, xmax, ymin, ymax,
                                                    polydeg, Nx, Ny)

  φ0 = zero(x)
  q1 = similar(φ0)
  q2 = similar(φ0)
  q = (q1, q2)
  f  = f_func.(x, y)
  tmp = similar(φ0)
  φsol  = φ_func.(x, y)
  q1sol = q1_func.(x, y)
  q2sol = q2_func.(x, y)
  φBC_left   = φ_func.(x[1,   :, 1,   :], y[1,   :, 1,   :])
  φBC_right  = φ_func.(x[end, :, end, :], y[end, :, end, :])
  φBC_bottom = φ_func.(x[:, 1,   :, 1, ], y[:, 1,   :, 1, ])
  φBC_top    = φ_func.(x[:, end, :, end], y[:, end, :, end])
  params = (; basis, dl, dr, Tᵣ, jacobian_x, jacobian_y, x, y,
              surface_values_x, surface_values_y,
              f, q, φBC_left, φBC_right, φBC_bottom, φBC_top, tmp,
              φsol, q1sol, q2sol)

  # # using a sparse direct solver is slow if we compute the sparse matrix
  # # via many matrix-vector products
  # A, b = compute_sparse_linear_structure(hypdiff_parabolic!, copy(φ0), params, 0.0)
  # φ = reshape(A \ b, size(φ0))

  # using CG is more efficient if we have only cheap matrix-vector products
  b = zeros(length(φ0))
  hypdiff_parabolic!(reshape(b, size(φ0)), φ0, params, 0.0)
  @. b = -b
  mul_by_mass_matrix!(reshape(b, size(φ0)), params)
  A = LinearOperator(eltype(b), length(b), length(b), true, true,
                     (dφ, φ) -> hypdiff_parabolic_for_cg!(dφ, φ, params))
  # use slightly stricter relative tolerance than the default
  # sqrt(eps(eltype(b)) to get full accuracy for higher grid resolutions
  φ_vector, stats = Krylov.cg(A, b; rtol = 1.0e-12)
  # @show stats
  φ = reshape(φ_vector, size(φ0))

  q1 = similar(φ)
  q2 = similar(φ)
  q = (q1, q2)
  compute_q!(q, φ, params)

  error_φ  = l2_norm(φ,  φsol,  params)
  error_q1 = l2_norm(q1, q1sol, params)
  error_q2 = l2_norm(q2, q2sol, params)
  return (; error_φ, error_q1, error_q2,
            x, y, φ, q1, q2, φsol, q1sol, q2sol, params)
end

function solve_2d_periodic(; polydeg = 3, N = 10, Nx = N, Ny = N)
  xmin, xmax = -1.0, 1.0
  ymin, ymax = -1.0, 1.0

  φ_func(x, y)  =      2 * cospi(x) * sinpi(2 * y)
  q1_func(x, y) = -π * 2 * sinpi(x) * sinpi(2 * y)
  q2_func(x, y) =  π * 4 * cospi(x) * cospi(2 * y)
  f_func(x, y)  = π^2*10 * cospi(x) * sinpi(2 * y)

  (; basis, dl, dr, Tᵣ, jacobian_x, jacobian_y, x, y,
     surface_values_x, surface_values_y) = setup_2d(xmin, xmax, ymin, ymax,
                                                    polydeg, Nx, Ny)

  φ0 = zero(x)
  q1 = similar(φ0)
  q2 = similar(φ0)
  q = (q1, q2)
  f  = f_func.(x, y)
  tmp = similar(φ0)
  φsol  = φ_func.(x, y)
  q1sol = q1_func.(x, y)
  q2sol = q2_func.(x, y)
  φBC_left   = :periodic
  φBC_right  = :periodic
  φBC_bottom = :periodic
  φBC_top    = :periodic
  params = (; basis, dl, dr, Tᵣ, jacobian_x, jacobian_y, x, y,
              surface_values_x, surface_values_y,
              f, q, φBC_left, φBC_right, φBC_bottom, φBC_top, tmp,
              φsol, q1sol, q2sol)

  b = zeros(length(φ0))
  hypdiff_parabolic!(reshape(b, size(φ0)), φ0, params, 0.0)
  @. b = -b
  mul_by_mass_matrix!(reshape(b, size(φ0)), params)
  A = LinearOperator(eltype(b), length(b), length(b), true, true,
                     (dφ, φ) -> hypdiff_parabolic_for_cg!(dφ, φ, params))
  # use slightly stricter relative tolerance than the default
  # sqrt(eps(eltype(b)) to get full accuracy for higher grid resolutions
  φ_vector, stats = Krylov.cg(A, b; rtol = 1.0e-12)
  # @show stats
  φ = reshape(φ_vector, size(φ0))

  q1 = similar(φ)
  q2 = similar(φ)
  q = (q1, q2)
  compute_q!(q, φ, params)

  error_φ  = l2_norm(φ,  φsol,  params)
  error_q1 = l2_norm(q1, q1sol, params)
  error_q2 = l2_norm(q2, q2sol, params)
  return (; error_φ, error_q1, error_q2,
            x, y, φ, q1, q2, φsol, q1sol, q2sol, params)
end


@inline function dirichlet_boundary_flux(u_inner, u_boundary, orientation, direction, equations::HyperbolicDiffusionEquations2D)
  # Calculate boundary flux by setting the value of the incoming characteristic
  # variables to the outgoing characteristic variables plus the boundary value.
  # This version is made to impose a BC on phi only.
  phi_bc, q1_bc, q2_bc = u_boundary
  phi, q1, q2 = u_inner
  inv_Tr = equations.inv_Tr
  sqrt_inv_Tr = sqrt(inv_Tr)
  # Curved version using the `orientation`, which is a vector
  if direction == 1 # -x
    flux = SVector(-q1 - sqrt_inv_Tr * (phi - phi_bc),
                   -inv_Tr * phi_bc,
                   zero(phi_bc)) * orientation[1]
  elseif direction == 2 # +x
    flux = SVector(-q1 + sqrt_inv_Tr * (phi - phi_bc),
                   -inv_Tr * phi_bc,
                   zero(phi_bc)) * orientation[1]
  elseif direction == 3 # -y
    flux = SVector(-q2 - sqrt_inv_Tr * (phi - phi_bc),
                   zero(phi_bc),
                   -inv_Tr * phi_bc) * orientation[2]
  else #if direction == 4 # +y
    flux = SVector(-q2 + sqrt_inv_Tr * (phi - phi_bc),
                   zero(phi_bc),
                   -inv_Tr * phi_bc) * orientation[2]
  end

  return flux
end

function solve_2d_nonperiodic_trixi(; polydeg = 3, N = 10, Nx = N, Ny = N,
                                      surface_flux = flux_godunov)
  # setup of Cockburn, Guzman, Wang (2008)
  xmin, xmax = -0.5, 0.5
  ymin, ymax = -0.5, 0.5
  φ_func(x, y)  =           cospi(x) * cospi(y)
  q1_func(x, y) =    -π   * sinpi(x) * cospi(y)
  q2_func(x, y) =    -π   * cospi(x) * sinpi(y)
  f_func(x, y)  = 2 * π^2 * cospi(x) * cospi(y)

  initial_condition(x, t, equations) = if iszero(t)
    return SVector(zero(t), zero(t), zero(t))
  else
    return SVector(φ_func(x[1], x[2]), q1_func(x[1], x[2]), q2_func(x[1], x[2]))
  end
  source_terms(u, x, t, equations) = let
    harmonic = source_terms_harmonic(u, x, t, equations)
    return SVector(f_func(x[1], x[2]), 0, 0) + harmonic
  end
  boundary_condition_nonperiodic(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations) = let
    u_boundary = initial_condition(x, one(t), equations)
    flux = dirichlet_boundary_flux(u_inner, u_boundary, orientation, direction, equations)
    return flux
  end

  equations = HyperbolicDiffusionEquations2D(Lr =
    (xmax - xmin) * (ymax - ymin) / (2π * sqrt((xmax - xmin)^2 + (ymax - ymin)^2)))
  solver = DGSEM(; polydeg, surface_flux)
  mesh = StructuredMesh((Nx, Ny), (xmin, ymin), (xmax, ymax), periodicity=(false, false))
  semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition, solver;
    source_terms, boundary_conditions = (
      x_neg = boundary_condition_nonperiodic,
      x_pos = boundary_condition_nonperiodic,
      y_neg = boundary_condition_nonperiodic,
      y_pos = boundary_condition_nonperiodic,))

  ode = semidiscretize(semi, (0.0, 1.0e2))
  steady_state_callback = SteadyStateCallback(abstol = 5.0e-12, reltol = 0.0)
  stepsize_callback = StepsizeCallback(cfl = 1.5)
  callbacks = CallbackSet(steady_state_callback, stepsize_callback)
  sol = redirect_stderr(devnull) do
    # avoid cluttering the REPL with output of the steady_state_callback
    Trixi.solve(
      ode, Trixi.HypDiffN3Erk3Sstar52(),
      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
      save_everystep = false, callback = callbacks)
  end

  usol = Trixi.compute_coefficients(initial_condition, 1.0, semi)
  diff = sol.u[end] - usol
  error_φ  = integrate(abs2 ∘ ((u, equations) -> u[1]), diff, semi; normalize = false) |> sqrt
  error_q1 = integrate(abs2 ∘ ((u, equations) -> u[2]), diff, semi; normalize = false) |> sqrt
  error_q2 = integrate(abs2 ∘ ((u, equations) -> u[3]), diff, semi; normalize = false) |> sqrt
  return (; error_φ, error_q1, error_q2, sol, usol, semi)
end

function solve_2d_periodic_trixi(; polydeg = 3, N = 10, Nx = N, Ny = N,
                                   surface_flux = flux_godunov)
  xmin, xmax = -1.0, 1.0
  ymin, ymax = -1.0, 1.0

  φ_func(x, y)  =      2 * cospi(x) * sinpi(2 * y)
  q1_func(x, y) = -π * 2 * sinpi(x) * sinpi(2 * y)
  q2_func(x, y) =  π * 4 * cospi(x) * cospi(2 * y)
  f_func(x, y)  = π^2*10 * cospi(x) * sinpi(2 * y)

  initial_condition(x, t, equations) = if iszero(t)
    return SVector(zero(t), zero(t), zero(t))
  else
    return SVector(φ_func(x[1], x[2]), q1_func(x[1], x[2]), q2_func(x[1], x[2]))
  end
  source_terms(u, x, t, equations) = let
    harmonic = source_terms_harmonic(u, x, t, equations)
    return SVector(f_func(x[1], x[2]), 0, 0) + harmonic
  end

  equations = HyperbolicDiffusionEquations2D(Lr =
    (xmax - xmin) * (ymax - ymin) / (2π * sqrt((xmax - xmin)^2 + (ymax - ymin)^2)))
  solver = DGSEM(; polydeg, surface_flux)
  mesh = StructuredMesh((Nx, Ny), (xmin, ymin), (xmax, ymax), periodicity=(true, true))
  semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition, solver;
    source_terms, boundary_conditions = boundary_condition_periodic)

  ode = semidiscretize(semi, (0.0, 1.0e2))
  steady_state_callback = SteadyStateCallback(abstol = 5.0e-12, reltol = 0.0)
  stepsize_callback = StepsizeCallback(cfl = 1.5)
  callbacks = CallbackSet(steady_state_callback, stepsize_callback)
  sol = redirect_stderr(devnull) do
    # avoid cluttering the REPL with output of the steady_state_callback
    Trixi.solve(
      ode, Trixi.HypDiffN3Erk3Sstar52(),
      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
      save_everystep = false, callback = callbacks)
  end

  usol = Trixi.compute_coefficients(initial_condition, 1.0, semi)
  diff = sol.u[end] - usol
  error_φ  = integrate(abs2 ∘ ((u, equations) -> u[1]), diff, semi; normalize = false) |> sqrt
  error_q1 = integrate(abs2 ∘ ((u, equations) -> u[2]), diff, semi; normalize = false) |> sqrt
  error_q2 = integrate(abs2 ∘ ((u, equations) -> u[3]), diff, semi; normalize = false) |> sqrt
  return (; error_φ, error_q1, error_q2, sol, usol, semi)
end


# 1D functions
function convergence_tests_1d(solve, polydeg, Ns; latex = false)
  errors_φ = Vector{Float64}()
  errors_q = Vector{Float64}()

  for N in Ns
    res = solve(; polydeg, N)
    push!(errors_φ, res.error_φ)
    push!(errors_q, res.error_q)
  end
  eoc_φ = compute_eoc(Ns, errors_φ)
  eoc_q = compute_eoc(Ns, errors_q)

  # print results
  data = hcat(Ns, errors_φ, eoc_φ, errors_q, eoc_q)
  header = ["N", "L2 error φ", "L2 EOC φ", "L2 error q", "L2 EOC q"]
  kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                  ft_printf("%.2e", [2, 4]),
                                  ft_printf("%.2f", [3, 5])))
  pretty_table(data; kwargs...)
  if latex
    pretty_table(data; kwargs..., backend=Val(:latex))
  end

  return nothing
end

function convergence_tests_1d(; kwargs...)
  Ns = 10 .* 2 .^ (0:6)

  for polydeg in 2:3
    println()
    @info "Convergence tests 1D nonperiodic" polydeg
    convergence_tests_1d(solve_1d_nonperiodic, polydeg, Ns; kwargs...)

    println()
    @info "Convergence tests 1D nonperiodic with Trixi.jl" polydeg
    convergence_tests_1d(solve_1d_nonperiodic_trixi, polydeg, Ns[1:end-2]; kwargs...)
  end

  for polydeg in 2:3
    println()
    @info "Convergence tests 1D periodic" polydeg
    convergence_tests_1d(solve_1d_periodic, polydeg, Ns; kwargs...)

    println()
    @info "Convergence tests 1D periodic with Trixi.jl" polydeg
    convergence_tests_1d(solve_1d_periodic_trixi, polydeg, Ns[1:end-2]; kwargs...)
  end

  return nothing
end


# 2D functions
function convergence_tests_2d(solve, polydeg, Ns; latex = false)
  errors_φ  = Vector{Float64}()
  errors_q1 = Vector{Float64}()
  errors_q2 = Vector{Float64}()

  for N in Ns
    res = solve(; polydeg, N)
    push!(errors_φ,  res.error_φ)
    push!(errors_q1, res.error_q1)
    push!(errors_q2, res.error_q2)
  end
  eoc_φ  = compute_eoc(Ns, errors_φ)
  eoc_q1 = compute_eoc(Ns, errors_q1)
  eoc_q2 = compute_eoc(Ns, errors_q2)

  # print results
  data = hcat(Ns, errors_φ, eoc_φ, errors_q1, eoc_q1, errors_q2, eoc_q2)
  header = ["N", "L2 error φ", "L2 EOC φ", "L2 error q1", "L2 EOC q1",
                                           "L2 error q2", "L2 EOC q2"]
  kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                  ft_printf("%.2e", [2, 4, 6]),
                                  ft_printf("%.2f", [3, 5, 7])))
  pretty_table(data; kwargs...)
  if latex
    pretty_table(data; kwargs..., backend=Val(:latex))
  end

  return nothing
end

function convergence_tests_2d(; kwargs...)
  Ns = 4 .* 2 .^ (0:4)

  for polydeg in 2:3
    println()
    @info "Convergence tests 2D nonperiodic" polydeg
    convergence_tests_2d(solve_2d_nonperiodic, polydeg, Ns; kwargs...)
  end

  for polydeg in 2:3
    println()
    @info "Convergence tests 2D periodic" polydeg
    convergence_tests_2d(solve_2d_periodic, polydeg, Ns; kwargs...)
  end

  return nothing
end
