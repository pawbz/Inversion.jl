

@userplot PlotAM

@recipe function f(p::PlotAM)
	itr=p.args[1]
	fvec=p.args[2]
	frate=p.args[3]
	nop=length(fvec)

	layout := (2,nop)

	for iop in 1:nop
		@series begin
			subplot := iop
			legend := false
			xscale := :log10
			yscale := :log10
			xlabel := "Iteration"
			seriestype := :scatter
			title := string("op ",iop)
			[itr], [fvec[iop]]
		end
		@series begin
			subplot := (1+round(Int,nop/2))+iop
			xscale := :log10
			yscale := :log10
			xlabel := "Iteration"
			legend := false
			title := string("op rate",iop)
			seriestype := :scatter
			[itr], [frate[iop]]
		end

	end


end


