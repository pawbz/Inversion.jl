
using Inversion
using Optim
using BenchmarkTools
using Misfits
using Base.Test
using Calculus

xx=Inversion.X(1000,5)

@btime Inversion.update_x!(xx) 
@btime Inversion.update_pI!(xx) 
@btime Inversion.update_gx!(xx) 
@btime Inversion.initialize!(xx) 




3333czz

x=randn(100)
y1=randn(100);
y2=randn(100).*100.;

f1(x)=Misfits.error_squared_euclidean!(nothing, x, y1, nothing)
f2(x)=Misfits.error_squared_euclidean!(nothing, x, y2, nothing)

g1(st,x)=Misfits.error_squared_euclidean!(st, x, y1, nothing)
g2(st,x)=Misfits.error_squared_euclidean!(st, x, y2, nothing)

optim_func=[f1,f2]
optim_grad=[g1,g2]

pa=Inversion.ParamMO(noptim=2,x_init=randn(size(x)),
    optim_func=optim_func,optim_grad=optim_grad)


println("****** NO MEMORY ALLOCATION ALLOWED HERE ******")
@btime pa.func(x, pa)

st=similar(x)
println("****** NO MEMORY ALLOCATION ALLOWED HERE ******")
@btime pa.grad!(st, x, pa)

f=x -> pa.func(x, pa)
g=(storage, x) -> pa.grad!(storage, x, pa)

# only first objective
pa.αvec=[1., 0.]
res=optimize(f,g, x, )
y11=Optim.minimizer(res)
@test y11 ≈  y1

# only second objective
pa.αvec=[0., 1.]
res=optimize(f,g, x, )
y22=Optim.minimizer(res)
@test y22 ≈ y2


# test finite difference gradient
for i in 1:3
    randn!(pa.αvec) # test for different weighting paramters
    g=Calculus.gradient(x -> pa.func(x,pa), x)
    pa.grad!(st, x, pa)
    @test st ≈ g
end
