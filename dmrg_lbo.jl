#LBO directly with RDM #This version works
using ITensors, ITensorMPS
using Adapt: adapt
using KrylovKit: eigsolve
using NDTensors: scalartype, timer
using Printf: @printf
using TupleTools: TupleTools

function permute(
  M::AbstractMPS, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)}
)::typeof(M)
  M̃ = typeof(M)(length(M))
  for n in 1:length(M)
    lₙ₋₁ = linkind(M, n - 1)
    lₙ = linkind(M, n)
    s⃗ₙ = TupleTools.sort(Tuple(siteinds(M, n)); by=plev)
    M̃[n] = ITensors.permute(M[n], filter(!isnothing, (lₙ₋₁, s⃗ₙ..., lₙ)))
  end
  set_ortho_lims!(M̃, ortho_lims(M))
  return M̃
end

function dmrg_lbo(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
  ITensorMPS.check_hascommoninds(siteinds, H, psi0)
  ITensorMPS.check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg_lbo(PH, psi0, sweeps; kwargs...)
end

function dmrg_lbo(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)
  for H in Hs
    ITensorMPS.check_hascommoninds(siteinds, H, psi0)
    ITensorMPS.check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHS = ProjMPOSum(Hs)
  return dmrg_lbo(PHS, psi0, sweeps; kwargs...)
end

function dmrg_lbo(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight=true, kwargs...)
  ITensorMPS.check_hascommoninds(siteinds, H, psi0)
  ITensorMPS.check_hascommoninds(siteinds, H, psi0')
  for M in Ms
    ITensorMPS.check_hascommoninds(siteinds, M, psi0)
  end
  H = permute(H, (linkind, siteinds, linkind))
  Ms .= permute.(Ms, Ref((linkind, siteinds, linkind)))
  if weight <= 0
    error(
      "weight parameter should be > 0.0 in call to excited-state dmrg_lbo (value passed was weight=$weight)",
    )
  end
  PMM = ProjMPO_MPS(H, Ms; weight)
  return dmrg_lbo(PMM, psi0, sweeps; kwargs...)
end

using NDTensors.TypeParameterAccessors: unwrap_array_type

# Get the left indices of a tensor 
function get_Linds(O::ITensor, tag::String)
    Rind = getfirst(x -> hastags(x, tag), inds(O))
    return noncommoninds([Rind], inds(O))
end

function dmrg_lbo(
  PH,
  psi0::MPS,
  sweeps::Sweeps;
  which_decomp=nothing,
  svd_alg=nothing,
  observer=NoObserver(),
  outputlevel=1,
  write_when_maxdim_exceeds=nothing,
  write_path=tempdir(),
  # eigsolve kwargs
  eigsolve_tol=1e-14,
  eigsolve_krylovdim=3,
  eigsolve_maxiter=1,
  eigsolve_verbosity=0,
  eigsolve_which_eigenvalue=:SR,
  ishermitian=true,
  LBO::Bool = true,
  max_LBO_dim=[6,4],
  min_LBO_dim::Int=4
  )
  if length(psi0) == 1
    error(
      "`dmrg_lbo` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  ITensors.@debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  psi = copy(psi0)
  N = length(psi)
  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end
  PH = position!(PH, psi, 1)
  energy = 0.0

  #If doing LBO, initialize the set of Rs
  if LBO
      sites = siteinds(psi)
      Rs = [ITensor() for _ in 1:N-1]
      #Rs = [ITensor() for _ in 1:round(Int,N/2)]
      PH_original = copy(PH)
      position!(PH_original, psi, 1)
  end

  
  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
        maxtruncerr = 0.0

        if !isnothing(write_when_maxdim_exceeds) &&
            maxdim(sweeps, sw) > write_when_maxdim_exceeds
            if outputlevel >= 2
              println(
                "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
              )
            end
                
            PH = disk(PH; path=write_path)
        end
            
        for (b, ha) in sweepnext(N)
            ITensors.@debug_check begin
                checkflux(psi)
                checkflux(PH)
            end

            """
            Optionally perform local basis optimization 
            """
                
            # NOTE: START WITH LARGER LBO_DIM and decrease with each sweep!
            #we optimize only boson sites 

            sw_warm = 2 #warmup sweeps, no lbo 
            if LBO && sw_warm+1+length(max_LBO_dim)>sw>sw_warm && iseven(b) && ha==1  #hastags(psi[b],"Boson")
                # Put into canonical form
                orthogonalize!(psi,b)
                M = psi[b] 
                H = PH.H[b] # we start fresh each time
                
                if hastags(M,"Trunc")
                    M = dag(Rs[b])*M
                    H = dag(prime(Rs[b]))*H*Rs[b]
                end
      
                rdm_b = prime(M,sites[b])*dag(M)
        
                U,S,V = svd(rdm_b, noprime(inds(rdm_b))[2], lefttags="Trunc", maxdim=max_LBO_dim[sw-sw_warm])
		if b==4 println("max LBO dim for sweep $(sw) : ", max_LBO_dim[sw-sw_warm])  end
                    
                Rs[b] = U
                M̃ = U*M 
                H̃ = dag(U)*H*prime(U)
                t = getfirst(x -> hastags(x, "Trunc"), inds(U))
                
                if dim(t) >= min_LBO_dim
                    PH.H[b] = H̃
                    psi[b] = M̃
                end
                
            end  #LBO ends
            # Normalize? 
            #normalize!(psi)
            
            ITensors.@timeit_debug timer "dmrg_lbo: position!" begin
              PH = position!(PH, psi, b)
            end

            ITensors.@debug_check begin
              checkflux(psi)
              checkflux(PH)
            end

            ITensors.@timeit_debug timer "dmrg_lbo: psi[b]*psi[b+1]" begin
              phi = psi[b] * psi[b + 1]
            end

            ITensors.@timeit_debug timer "dmrg_lbo: eigsolve" begin
              vals, vecs = eigsolve(
                PH,
                phi,
                1,
                eigsolve_which_eigenvalue;
                ishermitian,
                tol=eigsolve_tol,
                krylovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
              )
            end
                
            energy = vals[1]
            
            phi = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
              adapt(ITensors.set_eltype(unwrap_array_type(phi), eltype(vecs[1])), vecs[1])
            else
              vecs[1]
            end

            ortho = ha == 1 ? "left" : "right"

            drho = nothing
            if noise(sweeps, sw) > 0
              ITensors.@timeit_debug timer "dmrg_lbo: noiseterm" begin
                # Use noise term when determining new MPS basis.
                # This is used to preserve the element type of the MPS.
                elt = real(scalartype(psi))
                drho = elt(noise(sweeps, sw)) * noiseterm(PH, phi, ortho)
              end
            end

            ITensors.@debug_check begin
              checkflux(phi)
            end

            ITensors.@timeit_debug timer "dmrg_lbo: replacebond!" begin
              spec = replacebond!(
                PH,
                psi,
                b,
                phi;
                maxdim=maxdim(sweeps, sw),
                mindim=mindim(sweeps, sw),
                cutoff=cutoff(sweeps, sw),
                eigen_perturbation=drho,
                ortho,
                normalize=true,
                which_decomp,
                svd_alg,
              )
            end
            maxtruncerr = max(maxtruncerr, spec.truncerr)

            ITensors.@debug_check begin
              checkflux(psi)
              checkflux(PH)
            end

            if outputlevel >= 2
              @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
              @printf(
                "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                cutoff(sweeps, sw),
                maxdim(sweeps, sw),
                mindim(sweeps, sw)
              )
              @printf(
                "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
              )
              flush(stdout)
            end

            sweep_is_done = (b == 1 && ha == 2)
            measure!(
              observer;
              energy,
              psi,
              projected_operator=PH,
              bond=b,
              sweep=sw,
              half_sweep=ha,
              spec,
              outputlevel,
              sweep_is_done,
            )
        end #sweepnext ends
    
    end 
        
    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    isdone && break
        
  end #sweep ends

  # Transform back to original basis if implementing LBO
  if LBO
        for b in 2:2:length(Rs)
            psi[b] *= dag(Rs[b]) # Transform back into original basis 
        end
        
        return energy, psi
  end
  
  return (energy, psi);
end

function _dmrg_lbo_sweeps(;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=ITensorMPS.default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
)
  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, maxdim...)
  setmindim!(sweeps, mindim...)
  setcutoff!(sweeps, cutoff...)
  setnoise!(sweeps, noise...)
  return sweeps
end

function dmrg_lbo(
  x1,
  x2,
  psi0::MPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=ITensorMPS.default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  kwargs...,
)
  return dmrg_lbo(
    x1, x2, psi0, _dmrg_lbo_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...
  )
end

function dmrg_lbo(
  x1,
  psi0::MPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=ITensorMPS.default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  kwargs...,
)
  return dmrg_lbo(x1, psi0, _dmrg_lbo_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...)
end
