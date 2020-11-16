### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ ac900954-2445-11eb-238e-ffc1eae45120
begin
	using Turing
	using CSV, DataFrames, Distributions
	using MCMCChains, StatsFuns, StatsPlots, Plots
	using FillArrays
	using ReverseDiff
	using Zygote
end

# ╔═╡ f531c08a-25b2-11eb-25f9-a10f84850ba7
# Turing.setadbackend(:forwarddiff)
Turing.setadbackend(:reversediff)

# ╔═╡ c2b7bccc-2445-11eb-2dc8-37cd864a32b4
begin
	column_names = ["party", "smart", "creative", "hw", "mac", "project", "success", "happy"]

	df = CSV.read(joinpath(@__DIR__, "students.csv"), DataFrame, header=column_names,  delim=',';)

	# convert to bool
	df = df.!=(0)
end

# ╔═╡ 2ad98110-2837-11eb-308e-d37f265fbcb3
hw = Vector{Bool}(undef, 1)

# ╔═╡ cf39605e-2445-11eb-17c2-cd83ae2f8428
@model function happiness(party, smart, creative, hw, mac, project, success, happy) 
	if happy === nothing
		n = 1
		happy = Vector(undef, n)
	else
		n = size(happy)[1]
	end
	if hw === nothing
		hw = Vector(undef, n)
	end
	if mac === nothing
		mac = Vector(undef, n)
	end
	if project === nothing
		project = Vector(undef, n)
	end
	if success === nothing
		success = Vector(undef, n)
	end
	

    party_b ~ Beta(2,2)
	smart_b ~ Beta(2,2)
	creative_b ~ Beta(2,2)
	party ~ filldist(Bernoulli(party_b), n)
    smart ~ filldist(Bernoulli(smart_b), n)
    creative ~ filldist(Bernoulli(creative_b), n)
	
	# for each variable, we want to have a coefficient for each possible combination of cases. So everything has 2^2 cases, except for happy, which has 3 incoming arrows so has 2^3 cases.
	hw_coeff ~ filldist(Beta(2, 2), 4)
	mac_coeff ~ filldist(Beta(2, 2), 4)
	project_coeff ~ filldist(Beta(2, 2), 4)
	success_coeff ~ filldist(Beta(2, 2), 4)
	happy_coeff ~ filldist(Beta(2, 2), 8)
	

	for i = 1:n
		hw_idx = 2*party[i] + smart[i] .+ 1
		hw[i] ~ Bernoulli(hw_coeff[hw_idx])
		
		mac_idx = 2*creative[i] + smart[i] .+ 1
		mac[i] ~ Bernoulli(mac_coeff[mac_idx])
		
		project_idx = 2*creative[i] + smart[i] .+ 1
		project[i] ~ Bernoulli(project_coeff[project_idx])
		
		success_idx = 2*project[i] + hw[i] .+ 1
		success[i] ~ Bernoulli(success_coeff[success_idx])
		
		happy_idx = 4*success[i] + 2*mac[i] + party[i] .+ 1
		happy[i] ~ Bernoulli(happy_coeff[happy_idx])
	end
	
	
	return happy
end

# ╔═╡ b5ac3cd4-24bc-11eb-1675-c5041020abc5
df_reduced = df[1:1000, :]

# ╔═╡ 5fab8164-24bc-11eb-3c1e-55ca18586634
begin
	iterations = 200
	ϵ = 0.05
	τ = 10

	# test()()
	final_model = happiness(df["party"], df["smart"], df["creative"], df["hw"], df["mac"], df["project"], df["success"], df["happy"] )
	# chns1 = sample(final_model, HMC(ϵ, τ), iterations)
	chns1 = sample(final_model, NUTS(0.65), iterations)
	# chns2 = sample(final_model, SMC(), iterations)
end

# ╔═╡ 9c26bdcc-24bc-11eb-1ce9-014839c7dbc5
plot(chns1)

# ╔═╡ f1816d2e-24e2-11eb-3840-fdc78fb0790f
# this doesn't work, so we do need brackets
# prob"happy=1.0 | chain = chns1, model = final_model, creative=1.0, smart=1.0, party=1.0, project=1.0, mac=1.0, hw=1.0, success=1.0"

# ╔═╡ 2a2ab542-24e3-11eb-26a3-d357d783fd16
# Probability of being happy
prob"happy=[true] | chain = chns1, model = final_model, creative=nothing, smart=nothing, party=nothing, project=nothing, mac=nothing, hw=nothing, success=nothing"

# ╔═╡ b875bf6c-2750-11eb-33ac-411faf3424bc
# What is the probability of being happy given that you are smart and creative, value should be ~0.58132
prob"happy=[true] | chain = chns1, model = final_model, creative = [true], smart = [true], party = nothing, project = [true], mac = nothing, hw = [true], success = nothing"

# ╔═╡ 7224eda4-24bd-11eb-1b78-9f7767e10ce5
# this works, and returns a value around 0.38
prob"happy=[1] | chain = chns1, model = final_model, creative=[1], smart=[1], party=[1], project=[1], mac=[1], hw=[1], success=[1]"

# ╔═╡ 19c1ac62-24e3-11eb-2342-4fbd07281041
# this works, and returns a value around 0.31
prob"happy=[1.0] | chain = chns1, model = final_model, creative=[1.0], smart=[1.0], party=[1.0], project=[1.0], mac=[1.0], hw=[1.0], success=nothing"

# ╔═╡ ccb0700e-24c0-11eb-030e-0d1f6b4b287b
prob"mac=[1.0] | chain = chns1, model = final_model, creative=nothing, smart=nothing, party=nothing, project=nothing, happy=nothing, hw=nothing, success=nothing"

# ╔═╡ 844dcf30-24c2-11eb-2e2d-d3220534f2a3
prob"project=[true] | chain = chns1, model = final_model, creative=[0.0], smart=[0.0], party=[0.0], mac=[0.0], happy=[1.0], hw=[0.0], success=[0.0]"

# ╔═╡ 15cdb308-25a4-11eb-3592-3d363066ded7
prob"happy=[1.0] | chain = chns1, model = final_model, creative=[0.0], smart=[1.0], party=[1.0], project=nothing, mac=nothing, hw=nothing, success=nothing"

# ╔═╡ 86e8b916-25ae-11eb-113f-5b27b75b67c6
# 1.5
# This correctly gives the probability of P(party=T)~=0.60216 
prob"party=[1.0] | chain = chns1, model = final_model, happy=[1.0], creative=nothing, smart=nothing, project=nothing, mac=nothing, hw=nothing, success=nothing"

# ╔═╡ f9fdf5b6-24c2-11eb-370b-9da79928c34b
# 1.6
prob"mac=[true] | chain = chns1, model = final_model, creative=[1], smart=[1], party=nothing, project=nothing, happy=nothing, hw=nothing, success=nothing"

# ╔═╡ 9b16a6b0-24c2-11eb-2dd4-11f4189accdd
chns1[Symbol("mac_coefficients[1]")]

# ╔═╡ Cell order:
# ╠═ac900954-2445-11eb-238e-ffc1eae45120
# ╠═f531c08a-25b2-11eb-25f9-a10f84850ba7
# ╠═c2b7bccc-2445-11eb-2dc8-37cd864a32b4
# ╠═2ad98110-2837-11eb-308e-d37f265fbcb3
# ╠═cf39605e-2445-11eb-17c2-cd83ae2f8428
# ╠═b5ac3cd4-24bc-11eb-1675-c5041020abc5
# ╠═5fab8164-24bc-11eb-3c1e-55ca18586634
# ╠═9c26bdcc-24bc-11eb-1ce9-014839c7dbc5
# ╠═f1816d2e-24e2-11eb-3840-fdc78fb0790f
# ╠═2a2ab542-24e3-11eb-26a3-d357d783fd16
# ╠═b875bf6c-2750-11eb-33ac-411faf3424bc
# ╠═7224eda4-24bd-11eb-1b78-9f7767e10ce5
# ╠═19c1ac62-24e3-11eb-2342-4fbd07281041
# ╠═ccb0700e-24c0-11eb-030e-0d1f6b4b287b
# ╠═844dcf30-24c2-11eb-2e2d-d3220534f2a3
# ╠═15cdb308-25a4-11eb-3592-3d363066ded7
# ╠═86e8b916-25ae-11eb-113f-5b27b75b67c6
# ╠═f9fdf5b6-24c2-11eb-370b-9da79928c34b
# ╠═9b16a6b0-24c2-11eb-2dd4-11f4189accdd
