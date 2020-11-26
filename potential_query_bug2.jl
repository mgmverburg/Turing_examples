### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ bbd08030-2fe2-11eb-3a23-b5c049cefc03
using Turing


# ╔═╡ 03914758-2fe3-11eb-2685-5536efbb8ba8
@model function gdemo(x, y)
	println("START")
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    for i in 1:length(y)
        y[i] ~ Normal(x, sqrt(s))
    end
end



# ╔═╡ 065db94c-2fe3-11eb-3ecb-4dd70928fc27
begin
	model_gdemo = gdemo(1.0, [1.5, 0.0])
	c2 = sample(model_gdemo, NUTS(0.65), 100)

	result1 = prob"y = [1.5] | chain = c2, model = model_gdemo, x = missing"
end

# ╔═╡ Cell order:
# ╠═bbd08030-2fe2-11eb-3a23-b5c049cefc03
# ╠═03914758-2fe3-11eb-2685-5536efbb8ba8
# ╠═065db94c-2fe3-11eb-3ecb-4dd70928fc27
