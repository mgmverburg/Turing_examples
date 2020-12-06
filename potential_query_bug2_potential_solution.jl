### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 46b49414-371a-11eb-37cb-6fed4d714f79
using Turing



# ╔═╡ 4abd3142-371a-11eb-3981-bf6c962b0005
@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
	if (x === missing || x === nothing)
		x = rand(Normal(m, sqrt(s)), 1)[1]
	else
    	x ~ Normal(m, sqrt(s))
	end
	@show x
    for i in 1:length(y)
        y[i] ~ Normal(x, sqrt(s))
    end
end


# ╔═╡ 4c9055f0-371a-11eb-27fd-ed6604d78a47
begin
	model_gdemo = gdemo(1.0, [1.5, 0.0])
	c2 = sample(model_gdemo, NUTS(0.65), 100)

	
end

# ╔═╡ 4a5da5aa-371b-11eb-2566-11a6120fcc3f
result1 = prob"y = [1.5] | chain = c2, model = model_gdemo, x = missing"

# ╔═╡ Cell order:
# ╠═46b49414-371a-11eb-37cb-6fed4d714f79
# ╠═4abd3142-371a-11eb-3981-bf6c962b0005
# ╠═4c9055f0-371a-11eb-27fd-ed6604d78a47
# ╠═4a5da5aa-371b-11eb-2566-11a6120fcc3f
