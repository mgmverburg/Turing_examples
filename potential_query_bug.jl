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
	println("START")
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ filldist(Normal(m, sqrt(s)), length(y))
    @show x
    for i in 1:length(y)
        @show x[i]
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

# ╔═╡ Cell order:
# ╠═80ec1eaa-29a9-11eb-27db-ab3ebe3ed0ca
# ╠═a0c8f32e-29a9-11eb-102e-77e37bb2c0fd
# ╠═a5130988-29a9-11eb-2ba2-357bdcecce59
# ╠═bda140aa-29a9-11eb-35cc-cbbeb3398926
