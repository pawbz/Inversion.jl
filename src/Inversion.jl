"""
This module has different inversion schemes.
* Alternate Optimization
* Multi-objective Optimization
"""
module Inversion
using RecipesBase
using ProgressMeter
using DataFrames
using Misfits
using Printf

"""
Parameters for alternating minimization of a single objective function, while updating different model parameters
"""
mutable struct ParamAM
	name::String			# prints name of the optimization
	max_roundtrips::Int64		# limit number of roundtrips
	max_reroundtrips::Int64		# there will be reroundtrips, if roundtrips don't converge 
	noptim::Int64 			# number of optimizations in every roundtrip
	optim_func::Vector{Function}	# optimization functions, call them to perform optimizations
	reinit_func::Function		# re-initialize function that is executed if roundtrips fail to converge
	after_reroundtrip_func::Function # function executed after each reroundtrip (use it to store erros)
	after_roundtrip_func::Function  # function to be executed after certain number of roundtrips
	fvec::Matrix{Float64}		# functionals
	fvec_init::Vector{Float64}	# 
	optim_tols::Vector{Float64}	# tolerance for each optimization
	roundtrip_tol::Float64		# tolerance for stopping roundtrips
	verbose::Bool
	log::DataFrames.DataFrame
end



function ParamAM(optim_func;
	       name="",
	       noptim=length(optim_func),
	       optim_tols=[1e-3 for iop in 1:noptim],
	       roundtrip_tol=1e-3,
	       min_roundtrips=10, # minimum roundtrips before checking rate of convergence
	       verbose=true,
	       reinit_func=x->randn(),
	       after_reroundtrip_func=x->randn(),
	       after_roundtrip_func=x->randn(),
	       max_reroundtrips=1, re_init_flag=true, max_roundtrips=1)

	fvec=zeros(noptim, 2)
	fvec_init=zeros(noptim)

	# print dataframe
	log=DataFrame([[] for i in 1:4*noptim+3],
		 vcat(
			[:trip],
			[Symbol(string("J",i)) for i in 1:noptim],
			[Symbol(string("δJ",i)) for i in 1:noptim],
			[Symbol(string("fcalls",i)) for i in 1:noptim],
			[Symbol(string("gcalls",i)) for i in 1:noptim],
			[Symbol(string("time",i)) for i in 1:noptim]
			))

	pa=ParamAM(name, max_roundtrips, max_reroundtrips, noptim, optim_func, 
	  reinit_func, after_reroundtrip_func,
	  after_roundtrip_func,
	  fvec, fvec_init, optim_tols, roundtrip_tol, verbose, log)


	return pa



end

"""
Perform
alternating optimizations, updating different model parameters, computing a
same objective functional
"""
function go(pa::ParamAM, io=stdout)

	reroundtrip_converge=false
	itrr=0

	pa.verbose && write(io,string(pa.name, "\t alternate optimization\n"))  
	rf=zeros(pa.noptim)


	while ((!reroundtrip_converge && itrr < pa.max_reroundtrips))
		itrr += 1
		pa.verbose && (itrr > 1) && write(io,string("failed to converge.. reintializing (",itrr,"/",pa.max_reroundtrips,")\n"))
		pa.verbose && write(io,"=========================================================================================\n")  

	
		fill!(pa.fvec,0.0)
		fill!(pa.fvec_init,0.0)

		# execute re-initialization function
		(itrr > 1) && pa.reinit_func(nothing)

		itr=0
		roundtrip_converge=false

		# print
		if(pa.verbose)
			write(io,@sprintf( "trip\t|"))
			for iop in 1:pa.noptim
				write(io,@sprintf( "\t\top %d\t(%0.1e)\t|",iop, pa.roundtrip_tol))
			end
#			@printf("\tvar(op) (%0.1e)\t",pa.roundtrip_tol)
			write(io,@sprintf( "\n"))
		end
		if(io ≠ stdout)
			prog = ProgressThresh(pa.roundtrip_tol, "Minimizing:")
		end
		while !roundtrip_converge && itr < pa.max_roundtrips


			itr += 1

			# optimizations in each roundtrip
			for iop in 1:pa.noptim
				name=string("op ",iop," in each roundtrip")
				pa.fvec[iop,2]=pa.fvec[iop,1]
				pa.fvec[iop,1]=pa.optim_func[iop](nothing)
			end


			# store functionals at the first roundtrip
			if(iszero(pa.fvec_init))
				for iop in 1:pa.noptim
					pa.fvec_init[iop]=pa.fvec[iop,1]
				end
			end

			# normalize func for each optim
			for iop in 1:pa.noptim
				pa.fvec[iop,1] /= pa.fvec_init[iop]
			end

			# compute the change in the functions
			for iop in 1:pa.noptim
				rf[iop]=abs(pa.fvec[iop,2]-pa.fvec[iop,1])/pa.fvec[iop,2]
			end

			#=
			if((2<itr<5) ||(itr<40 && (mod(itr,5)==0)) || ((itr<500 && mod(itr,20)==0)) && (mod(itr,100)==0))
				plotam!(itr,pa.fvec[:,1],rf)
			elseif(itr==2)
				plotam(itr,pa.fvec[:,1],rf)
			end
			=#
			(io ≠ stdout) && ProgressMeter.update!(prog, maximum(rf))

			if(itr > 10)# do atleast 10 round trips before quitting
				roundtrip_converge=all(rf .< pa.roundtrip_tol) || all(pa.fvec[:,1] .< pa.optim_tols[:])
			else
				roundtrip_converge=false
			end

			
			push!(pa.log[:trip], itr)
			for iop in 1:pa.noptim
				push!(pa.log[Symbol(string("J",iop))], pa.fvec[iop,1])
				if(itr>2)
					push!(pa.log[Symbol(string("δJ",iop))], rf[iop])
				end
			end

			# print info
			if(pa.verbose)
				if((itr<5) ||(itr<40 && (mod(itr,5)==0)) || (itr<500 && (mod(itr,20)==0)) || (mod(itr,50)==0) || roundtrip_converge)

					pa.after_roundtrip_func(nothing) # after each roundtrip, execute this
					#if(itr==1)
					#	show(pa.log)
					#else
					#	DataFrames.showrowindices(stdout, pa.log, [itr], 
					#			  DataFrames.getmaxwidths(pa.log, 1:4, 1:4, :Row), 1, 4)
					#end
					write(io,@sprintf( "%d\t|",itr))
					for iop in 1:pa.noptim
						push!(pa.log[Symbol(string("J",iop))], pa.fvec[iop,1])
						if(itr>2)
							push!(pa.log[Symbol(string("δJ",iop))], rf[iop])
						end
						write(io,@sprintf( "\t%0.6e\t",pa.fvec[iop,1]))
						(itr==1) ? write(io,@sprintf( "\t\t|")) : write(io,@sprintf( "(%0.6e)\t|",rf[iop]))
					end
					write(io,@sprintf( "\n"))
				end
				flush(io)
			end
			#show(pa.log)

			# variance b/w objectives
#			rf=vecnorm(pa.fvec[:,1])/vecnorm(fill(pa.fvec[1,1],pa.noptim))
#			if(pa.verbose)
#				@printf("\t%0.6e\t|",rf)
#				@printf("\n")
#			end
		end
		pa.after_reroundtrip_func(nothing) # after each roundtrip, execute this
		reroundtrip_converge = all(pa.fvec[:,1] .< pa.optim_tols[:])
		pa.verbose && write(io,"=========================================================================================\n")  
	end
	if(all(pa.fvec[:,1] .< pa.optim_tols[:]))
		message=string("CONVERGED in ",itrr," reroundtrips")
	elseif(itrr == pa.max_reroundtrips)
		message="NOT CONVERGED: reached maximum reroundtrips"
	end
	pa.verbose && write(io, string(pa.name, "\t", message, "\n")) 
	return nothing
end

include("X.jl")
include("plots.jl")
#=
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Multi-objective Inversion Framework

# Credits:
#	Pawan Bharadwaj
#	November 2017

"""
Type for multi-objective inversion variable
"""
type ParamMO
	noptim				# number of optimizations

	optim_func::Vector{Function}	# optimization functions
	optim_grad::Vector{Function} 	# gradient computing functions

	αvec::Vector{Float64}		# weights for each parameters 
	fvec_init::Vector{Float64}	# store initial functionals for scaling
	fvec::Vector{Float64}		# store functional in each call
	storage_temp::Vector{Float64}	# temp storage of gradient

	func::Function			# call this multi-objective functional
	grad!::Function			# call this multi-objective gradient 

end


"""
Scalarize a vector of multiple objective functions
* `x_init::Vector{Float64}` : model to compute initial functionals
* `optim_func::Vector{Function}` : to compute functionals
* `optim_grad::Vector{Function}` : to compute gradients

Only allocation is creating an additional vector for temporary gradient storage
"""
function ParamMO(;
		 noptim=1,
		 αvec=ones(noptim),
		 ninv=nothing, #number of inversion variables
		 x_init=nothing,
		 optim_func=nothing,
		 optim_grad=nothing,
		 )
	(length(optim_func) != noptim) && error("length func")
	(length(optim_grad) != noptim) && error("length grad")
	(length(αvec) != noptim) && error("length αvec")

	fvec=zeros(noptim)
	fvec_init=zeros(noptim)

	# allocate storage_temp
	if(!(ninv===nothing))
		storage_temp=zeros(ninv)
	elseif(!(x_init===nothing))
		storage_temp=zeros(x_init)
	else
		error("need ninv or x_init")
	end
	pa=ParamMO(noptim,optim_func,optim_grad,αvec,fvec_init,fvec,storage_temp,func,grad!)

	# compute functional using x_init and store them 
	if(!(x_init===nothing))
		update_fvec_init!(x_init, pa)
	end

	return pa
end

# func scalarizes fvec
function func(x, pa)
	f=0.0
	for iop in 1:pa.noptim
		pa.fvec[iop]=pa.optim_func[iop](x)/pa.fvec_init[iop]
		f += (pa.fvec[iop]*pa.αvec[iop])
	end
	return f
end

# sum gradients with proper weights
function grad!(storage, x, pa)
	f=0.0
	storage[:]=0.0
	for iop in 1:pa.noptim
		pa.fvec[iop]=pa.optim_grad[iop](pa.storage_temp, x)*inv(pa.fvec_init[iop])
		f += (pa.fvec[iop]*pa.αvec[iop])
		scale!(pa.storage_temp, pa.αvec[iop]*inv(pa.fvec_init[iop]))
		for i in eachindex(storage)
			storage[i] += pa.storage_temp[i]
		end
	end
	return f
end


"""
Average the reference values of function over second dimension of x
"""
function update_fvec_init!(x_init, pa::ParamMO)
	pa.fvec_init[:]=0.0
	avgsize=size(x_init,2)
	for iv in 1:avgsize
		xi=view(x_init,:,iv)
		for iop in 1:pa.noptim
			pa.fvec_init[iop]+=pa.optim_func[iop](xi)
		end
	end
	scale!(pa.fvec_init, inv(avgsize))
	any(iszero.(pa.fvec_init)) && error("fvec_init cannot be zero")
end


=#

end # module
