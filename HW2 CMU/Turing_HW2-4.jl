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
	using Logging: global_logger
	using TerminalLoggers: TerminalLogger
end

# ╔═╡ 6f80f0f8-2974-11eb-17f3-533d1e8cce49
global_logger(TerminalLogger())

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
@model function happiness(party, smart, creative, hw, mac, project, success, happy, ::Type{T} = Bool) where {T} 
	if happy === nothing || happy === missing
		n = 1
		happy = Vector{T}(undef, n)
	else
		n = size(happy)[1]
	end
	if hw === nothing || hw === missing
		hw = Vector{T}(undef, n)
	end
	if mac === nothing || mac === missing 
		mac = Vector{T}(undef, n)
	end
	if project === nothing || project === missing
		project = Vector{T}(undef, n)
	end
	if success === nothing || success === missing
		success = Vector{T}(undef, n)
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
# potentially reduce the size of the dataset in case runtime is too long. I try to keep the runtime around a few minutes usually.
df_red = df[1:5000, :]

# ╔═╡ 5fab8164-24bc-11eb-3c1e-55ca18586634
begin
	iterations = 200
	ϵ = 0.05
	τ = 10

	# test()()
	final_model = happiness(df_red["party"], df_red["smart"], df_red["creative"], df_red["hw"], df_red["mac"], df_red["project"], df_red["success"], df_red["happy"] )
	# final_model = happiness(missing, missing, missing, missing, missing, missing, missing, df_red["happy"] )
	# chns1 = sample(final_model, HMC(ϵ, τ), iterations)
	chns1 = sample(final_model, NUTS(0.65), iterations)
	# chns2 = sample(final_model, SMC(), iterations)
end

# ╔═╡ 9c26bdcc-24bc-11eb-1ce9-014839c7dbc5
plot(chns1)

# ╔═╡ 2a2ab542-24e3-11eb-26a3-d357d783fd16
# Probability of being happy
prob"happy=[true] | chain = chns1, model = final_model, creative=nothing, smart=nothing, party=nothing, project=nothing, mac=nothing, hw=nothing, success=nothing"

# ╔═╡ 0e92dd0e-28ae-11eb-14c6-77e4a23f66e3
begin
	creative_d = df_red["creative"]
	smart_d = df_red["smart"]
	party_d = df_red["party"]
	project_d = df_red["project"]
	mac_d = df_red["mac"]
	hw_d = df_red["hw"]
	success_d = df_red["success"]
	happy_d = fill(true, size(success_d)[1])
end

# ╔═╡ f2f70444-28ad-11eb-2f23-a9765c9cb332
# Probability of being happy
result = prob"happy=[true] | chain = chns1, model = final_model, creative=creative_d, smart=smart_d, party=party_d, project=project_d, mac=mac_d, hw=hw_d, success=success_d"

# ╔═╡ c640ec20-28b8-11eb-018d-2388c1fe3bba
# P(happy = T) = 0.51575
# Our found answer: 0.5164463373803578
begin
	target_happy = [true]
	local counter = 1
	result_prob = Vector{Float64}(undef, 128)
	options = [[true], [false]]
	for creative_v = options
		for smart_v = options
			for party_v = options
				for project_v = options
					for mac_v = options
						for hw_v = options
							for success_v = options
								result = prob"happy=target_happy, creative=creative_v, smart=smart_v, party=party_v, project=project_v, mac=mac_v, hw=hw_v, success=success_v | chain = chns1, model = final_model"
								result_prob[counter] = mean(result)
								counter += 1
								
							end
						end
					end
				end
			end
		end
	end
	
	# prob"happy=target_happy, creative=[true], smart=[true], party=[true], project=[true], mac=[true], hw=[true], success=[true] | chain = chns1, model = final_model"
end

# ╔═╡ 9560f272-28bc-11eb-0265-1fab4e4cc6b7
# this gives the answer we are looking for for P(happy=T)
sum(result_prob)

# ╔═╡ e205645e-28c2-11eb-373a-f1c104c3a119
begin
	local target_happy = [true]
	local party_v = [false]
	local hw_v = [true]
	local project_v = [true]
	local counter = 1
	result_prob_B_A = Vector{Float64}(undef, 16)
	local options = [[true], [false]]
	for creative_v = options
		for smart_v = options
			for mac_v = options
				for success_v = options
					result = prob"happy=target_happy, creative=creative_v, smart=smart_v, party=party_v, project=project_v, mac=mac_v, hw=hw_v, success=success_v | chain = chns1, model = final_model"
					result_prob_B_A[counter] = mean(result)
					counter += 1
								
				end
			end
		end
	end
end

# ╔═╡ 861a3ace-28c3-11eb-3147-99a358cde6ce
begin
	local party_v = [false]
	local hw_v = [true]
	local project_v = [true]
	local counter = 1
	result_prob_B = Vector{Float64}(undef, 32)
	local options = [[true], [false]]
	for creative_v = options
		for smart_v = options
			for mac_v = options
				for success_v = options
					for happy_v = options
						result = prob"happy=happy_v, creative=creative_v, smart=smart_v, party=party_v, project=project_v, mac=mac_v, hw=hw_v, success=success_v | chain = chns1, model = final_model"
						result_prob_B[counter] = mean(result)
						counter += 1
					end
				end
			end
		end
	end
end

# ╔═╡ 19b25f4c-28c3-11eb-2f45-4bdf52717421
# P(happy = T | party = F, hw = T, project = T) = 0.32108
# which is very close to our result of 0.32106783203508416
sum(result_prob_B_A)/sum(result_prob_B)
# this works because of bayes' rule P(A|B)=P(B,A)/P(B).

# ╔═╡ Cell order:
# ╠═ac900954-2445-11eb-238e-ffc1eae45120
# ╠═6f80f0f8-2974-11eb-17f3-533d1e8cce49
# ╠═f531c08a-25b2-11eb-25f9-a10f84850ba7
# ╠═c2b7bccc-2445-11eb-2dc8-37cd864a32b4
# ╠═2ad98110-2837-11eb-308e-d37f265fbcb3
# ╠═cf39605e-2445-11eb-17c2-cd83ae2f8428
# ╠═b5ac3cd4-24bc-11eb-1675-c5041020abc5
# ╠═5fab8164-24bc-11eb-3c1e-55ca18586634
# ╠═9c26bdcc-24bc-11eb-1ce9-014839c7dbc5
# ╠═2a2ab542-24e3-11eb-26a3-d357d783fd16
# ╠═0e92dd0e-28ae-11eb-14c6-77e4a23f66e3
# ╠═f2f70444-28ad-11eb-2f23-a9765c9cb332
# ╠═c640ec20-28b8-11eb-018d-2388c1fe3bba
# ╠═9560f272-28bc-11eb-0265-1fab4e4cc6b7
# ╠═e205645e-28c2-11eb-373a-f1c104c3a119
# ╠═861a3ace-28c3-11eb-3147-99a358cde6ce
# ╠═19b25f4c-28c3-11eb-2f45-4bdf52717421
