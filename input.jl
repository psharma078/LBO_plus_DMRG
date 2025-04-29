include("Hubb_exHolstein.jl")

#Hubbard para
Ns = 6
N = Ns*2   
U = 4.2 

Nup = 3#2
Ndn = 3#2


#phonon para
wo = 5.0   #omega
lamda = 3.6
g = sqrt(lamda*wo/2.0)

#initial phonon 
init_n = 2

#LBO cutoff
barePhononDim = 60 #80
LBO_dims = [40,30,40,20,10,16,10,8]
#LBO_dims = [80,70,60,50,50,40,30,20,12]

#boundary
pbc=false

@show Ns
@show U
@show Nup, Ndn
@show wo
@show lamda, g
@show init_n

gs_calcs(N,U,Nup,Ndn,wo,g,barePhononDim,LBO_dims,init_n,pbc);
