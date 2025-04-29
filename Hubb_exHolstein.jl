using ITensors, ITensorMPS, Plots, HDF5, Random
include("dmrg_lbo.jl")

#conserve electron qns without conserving phonons
function ITensors.space(
  ::SiteType"Qudit";
  dim=2,
  conserve_qns=false,
  conserve_number=conserve_qns,
  qnname_number="Number",
)
  if conserve_number
    return [QN(qnname_number, n - 1) => 1 for n in 1:dim]
  else
    return [QN() => dim]
  end
  return dim
end

function mixed_sites(N::Int,max_boson_dim::Int)
  sites = Vector{Index}(undef,N)
  for n=1:N
    if isodd(n)
      sites[n] = siteind("Electron"; addtags="n=$n", conserve_qns=true)
    else
      sites[n] = siteind("Boson"; addtags ="n=$n", dim = max_boson_dim, conserve_qns=false)
    end
  end
  return sites
end

function Hubbard_Holstein(N::Int64,U::Float64,ω::Float64,g::Float64, pbc::Bool=false)
    t = 1.0

    os = OpSum()

    for j in 1:2:(N-2)
    	os += -t, "Cdagup", j, "Cup", j+2
    	os += -t, "Cdagup", j+2, "Cup", j
    	os += -t, "Cdagdn", j, "Cdn", j+2
    	os += -t, "Cdagdn", j+2, "Cdn", j
    end
    #periodic boundary condition
    if pbc
        os += -t, "Cdagup", N-1, "Cup", 1
        os += -t, "Cdagup", 1, "Cup", N-1
        os += -t, "Cdagdn", N-1, "Cdn", 1
        os += -t, "Cdagdn", 1, "Cdn", N-1
    end

    #Hubbard U
    for j in 1:2:N-1
        os += U, "Nupdn", j
    end
 
    #on-site el-ph coupling  
    for j in 1:2:N-1
    	os += ω, "N", j+1
    	os += g, "Ntot", j, "A", j+1
    	os += g, "Ntot", j, "Adag", j+1
        os += -g, "A",j+1
        os += -g, "Adag",j+1
    end
 
    return os
end

function gs_calcs(N,U,Nup,Ndn,ω,g,barePhononDim,LBO_dims,init_n,pbc)
    os = Hubbard_Holstein(N, U, ω, g, pbc)
    
    sites = mixed_sites(N,barePhononDim)
    H = MPO(os,sites)

    ntot = Nup + Ndn  #init state for dopped system
    select_el_sites = sort(shuffle(collect(1:2:N))[1:ntot])
    state = [isodd(n) ? "Emp" : "$(init_n)" for n in 1:N]
    for (id,s) in enumerate(select_el_sites)
        state[s] = isodd(id) ? "Up" : "Dn"
    end
    @show state

    #= 
    #init state for half filling
    state = Vector{String}(undef, N)
    for xy = 1 : N
        if xy % 2 == 1    #elecreon site
            state[xy] = trunc(xy/2) % 2 == 0 ? "Up" : "Dn"
        else              #Boson site
            state[xy] = "$(init_n)"
        end
        
    end
    =#

    psi0 = MPS(sites,state)
    #psi0 = randomMPS(sites, state; linkdims = 10)
    @show expect(psi0,"N",sites=[n for n in 2:2:N])
    
    nsweeps = 20
    maxdim = [10,20,50,50,100,100,100,200,200,200,200,200,400,400,400,400,800,1500]
    cutoff = [1e-8]
    noise = [1E-2,1E-2,1E-2,1E-3,1E-3,1E-3,1E-4,1E-4,1e-4,1E-4,1E-5,1E-5,1e-6, 1e-6,1E-6,1E-6,1e-8,1e-8]
    #significant noise in the starting sweep is important. This is checked and verified.

    obs = DMRGObserver(; energy_tol=1e-10)
    #energy_bare, psi = dmrg(H,psi0;nsweeps=nsweeps,maxdim=maxdim,cutoff=cutoff,noise=noise,observer=obs,eigsolve_krylovdim= 8);
    #@show expect(psi, "N", sites = [n for n in 2:2:N])
    #println()
   
    energy_lbo, psi  = dmrg_lbo(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise, observer=obs,
    		                   eigsolve_krylovdim= 8, LBO=true, max_LBO_dim=LBO_dims,min_LBO_dim=2);
    println()
    Nboson = expect(psi,"N",sites=[n for n in 2:2:N])
    @show Nboson
    println()
    Sz = expect(psi,"Sz", sites=1:2:N)
    @show Sz
    
    return 0;
end
