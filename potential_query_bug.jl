### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 80ec1eaa-29a9-11eb-27db-ab3ebe3ed0ca
begin
	using Turing, Distributions
end

# ╔═╡ a0c8f32e-29a9-11eb-102e-77e37bb2c0fd
@model function gdemo(x, y)
	if x === missing || x === nothing
        # Initialize `x` if missing
        x = Vector{Float64}(undef, 2)
	end
	n = length(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ filldist(Normal(m, sqrt(s)), n)
	for i in 1:length(y)
    	y[i] ~ Normal(x[i], sqrt(s))
	end
end

# ╔═╡ a5130988-29a9-11eb-2ba2-357bdcecce59
begin
	model_gdemo = gdemo([1.0, 0.0], [1.5, 0.0])
	c2 = sample(model_gdemo, NUTS(0.65), 100)
end

# ╔═╡ bda140aa-29a9-11eb-35cc-cbbeb3398926
result1 = prob"y = [1.5] | chain=c2, model = model_gdemo, x = [1.0]"

# ╔═╡ c07b08ce-29a9-11eb-236f-050c9c7d8d56
mean(result1)

# ╔═╡ eb40ec90-29a9-11eb-3437-f3ecc0116310
result2 = prob"y = [1.5] | chain=c2, model = model_gdemo, x = [0.0]"

# ╔═╡ f3a02f7c-29a9-11eb-184f-e924eab6e213
mean(result2)

# ╔═╡ fbc6efe2-29a9-11eb-228e-6391f92a2c49
result3 = prob"y = [1.5] | chain=c2, model = model_gdemo, x = nothing"

# ╔═╡ 00d1d66e-29aa-11eb-2f02-e1d0ed936810
mean(result3)

# ╔═╡ 39e1ada8-29aa-11eb-08d0-b3c013decfb7
result4 = prob"y = [1.5], x = [1.0] | chain=c2, model = model_gdemo"

# ╔═╡ 43f0750e-29aa-11eb-0044-1f25f6e2d94f
mean(result4)

# ╔═╡ 4f2d1ce2-29aa-11eb-38ac-a7f44b4d530a
result5 = prob"y = [1.5], x = [0.0] | chain=c2, model = model_gdemo"

# ╔═╡ 5417da60-29aa-11eb-0123-03554284afab
mean(result5)

# ╔═╡ Cell order:
# ╠═80ec1eaa-29a9-11eb-27db-ab3ebe3ed0ca
# ╠═a0c8f32e-29a9-11eb-102e-77e37bb2c0fd
# ╠═a5130988-29a9-11eb-2ba2-357bdcecce59
# ╠═bda140aa-29a9-11eb-35cc-cbbeb3398926
# ╠═c07b08ce-29a9-11eb-236f-050c9c7d8d56
# ╠═eb40ec90-29a9-11eb-3437-f3ecc0116310
# ╠═f3a02f7c-29a9-11eb-184f-e924eab6e213
# ╠═fbc6efe2-29a9-11eb-228e-6391f92a2c49
# ╠═00d1d66e-29aa-11eb-2f02-e1d0ed936810
# ╠═39e1ada8-29aa-11eb-08d0-b3c013decfb7
# ╠═43f0750e-29aa-11eb-0044-1f25f6e2d94f
# ╠═4f2d1ce2-29aa-11eb-38ac-a7f44b4d530a
# ╠═5417da60-29aa-11eb-0123-03554284afab
