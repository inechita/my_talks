### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 03b1e4ea-d13f-4f2f-adc6-341598a30658
begin
	using PlutoUI
	using Random
	using LinearAlgebra
	using Plots
	using RandomMatrices
	PlutoUI.TableOfContents()
end

# ╔═╡ 04312c66-7d48-11ee-1562-83bf7dd613d8
md"""
# Random tensor flattenings
"""

# ╔═╡ 43b09095-6198-4544-8137-00fab6f5ff11
md"""
joint work with Stéphane Dartois and Camille Male, preprint [arXiv:2307.11439](https://arxiv.org/abs/2307.11439)
"""

# ╔═╡ 3a4824db-9674-45c1-90e4-c7a0c9cac464
md"""
## Problem statement
"""

# ╔═╡ 58903216-1306-4fc1-8e25-658c4a3b7435
md"""
We consider (random) complex tensors $M \in (\mathbb C^N)^{\otimes 2k}$. We have thus two parameters: 
- the Hilbert space dimension $N$, which we shall think of as being large; our goal is to compute the $N \to \infty$ limit of the model
- the order of the tensor $2k$; we consider only tensors of even order, for reasons which shall be clear in a moment
"""

# ╔═╡ b0b0a74b-f196-402b-a9d2-45758053542a
md"""
We shall make extensive use of the graphical notation for tensors due to [Penrose](https://en.wikipedia.org/wiki/Penrose_graphical_notation), where a tensor of order $2k$ is denoted by a box with $2k$ wires (legs) sticking out of it. If one attaches indices (from $\{1,2,\ldots, N\}$) to these legs, one gets the corresponding component $M_{i_1, \ldots, i_{2k}} \in \mathbb C$ of the tensor.

$(Resource("https://i.imgur.com/rVcMXJd.png", :width => 150))
"""

# ╔═╡ bcdde9e3-aebd-4795-80db-af957a15764a
md"""
In the simplest cases $k=1,2$, tensors have 2, respectively 4 legs:

![Imgur](https://i.imgur.com/Vk3kUyq.png)
"""

# ╔═╡ c5bda79a-c3df-45ad-bc6f-9e61dd155cff
md"""
### Flattenings
"""

# ╔═╡ 874575ff-0390-4bd4-a2dd-f9205e0c83ab
md"""

**Tensors are hard, matrices are easy.**

Flattenings (aka matricizations, views, etc...) are tranformations which map a tensor (think 3D cube, 3 indices) to matrices (think 2D array, 2 inidices). 

The $2k$ legs of a tensor are on equal footing; however, the legs of a matrix (linear operator) are of two types:
- inputs
- outputs

This fact can be also understood from the canonical isomorphism

$\mathcal M_{out \times in}(\mathbb C) \cong \mathbb C^{out} \otimes (\mathbb C^{in})^*.$
"""

# ╔═╡ 66e0d010-1508-46d4-bd24-0e203095342f
md"""
We shall be considering _balanced_ flattenings, where $k$ (out of the $2k$) legs will be input legs, and the other half will be output legs. This choice can be encoded by a two step procedure:
1. permute the $2k$ legs of the tensor $M$ using a permutation $\sigma \in \mathfrak S_{2k}$
2. decide that the top $k$ legs of the resulting tensor are _output_ legs and that the bottom $k$ legs are _input_ legs. 
![Imgur](https://i.imgur.com/cWjrKgr.png)
The resulting **matrix**, called the _$\sigma$-flattening_ of $M$, shall be denoted by

$M_\sigma \in \mathcal M_{N^k \times N^k}(\mathbb C).$
"""

# ╔═╡ b75ea572-c70b-4f36-836b-941350fb099f
md"""
For example, in the case $k=1$, a 2-tensor $M \in (\mathbb C^N)^{\otimes 2}$ has two flattenings $M_{(1)(2)}, M_{(12)} \in \mathcal M_N(\mathbb C)$:

![Imgur](https://i.imgur.com/D7gNjS9.png)

Remarkably, the two flattenings are transposes of eachother:

$M_{(12)} = \big( M_{(1)(2)} \big)^\top.$
We shall get back to this fact when discussing asymptotic freeness.
"""

# ╔═╡ 37aee748-f6bc-4421-9c37-e163308d71b9
md"""
Here is another important observation. Consider a pair permutations $\eta, \eta' \in \mathfrak S_k$ and the permutation $\eta \sqcup \eta' \in \mathfrak S_{2k}$ which acts like $\eta$ on the first $k$ elements and as $\eta'$ on the last $k$ elements of $[2k]$. We have the following relation: 

$M_{\eta \sqcup \eta'} = \eta \cdot M_{\mathrm{id}} \cdot (\eta')^{-1} = \eta \cdot M_{\mathrm{id}} \cdot (\eta')^*.$

With diagrams: 

![Imgur](https://i.imgur.com/kwboaJ9.png)

This means that tensor flattenings which correspond to a permutation which is a disjoint sum of two $k$-permutations are related to the identity flattening by _deterministic_ tensor-permutation matrices.
"""

# ╔═╡ b8f00bcb-0cf9-4795-9405-2d6381f0b4ff
md"""
### Random tensors
"""

# ╔═╡ 88fa8bb5-f7c4-4215-8430-666a899d4ad9
md"""
Let us now describe the model of random tensors that we are considering. The distribution of our random tensors $M$ is assumed to satisfy the following _Tensor-Distribution-Hypothesis_:
- the $N^{2k}$ (complex) entries of the tensor are _independent, identically distributed_ random variables
- their common distribution $\mu_N$ is _well behaved_, in the following sense: 

If $m_n \sim \mu_N$, then:
- the first moment is zero (tensor compoents are centered): $\mathbb E \big[ m_N \big] = 0$
- the second moments converge to a pair $(c,c')$:

$\begin{align}
\lim_{N \to \infty} N^k  \mathbb E\big[ |m_N|^2\big]  &=  c>0 \\
\lim_{N \to \infty} N^k  \mathbb E\big[ m_N^2 \big]   &= c' \in \mathbb C
\end{align}$
- higher moments behave as follows: for all non-negative integers $\ell_1,\ell_2$ such that $\ell_1+\ell_2>2$

$\lim_{N \to \infty} N^k \mathbb E \big[ m_N^{\ell_1} \cdot \overline{m_N}^{\ell_2}\big]  = 0.$

We call $(c,c')$ the _parameter of our model_.
"""

# ╔═╡ d3f708a7-4a0e-4779-bf93-16d2494f5976
md"""
The main examples we have in mind are the following. Let $M_N$ be sampled according to the _complex Ginibre ensemble_, i.e. the entries of $\sqrt N M_N$ are distributed according to the standard complex Gaussian distribution. Then $M_N$ satisfies the _Tensor-Distribution-Hypothesis_ and its parameter is $(1,0)$. A real Ginibre ensemble also satisfies the hypothesis with parameter $(1,1)$.
"""

# ╔═╡ 59ce6289-e929-4fdd-9bd3-447966b55691
md"""
In our paper, we provide a complete answer to the following question:

**Problem.** Describe the _joint distribution_ of the set of _all flattenings_

$\big\{ M_\sigma \big\}_{\sigma \in \mathfrak S_{2k}}$
for a random tensor $M$ satisfying the _Tensor-Distribution-Hypothesis_, in the large $N$ limit. 
"""

# ╔═╡ 530fa17a-61cf-4799-aac5-2ee82dfab4b5
md"""
It is important to stress that we are interested in the collection of _all_ the flattenings, in the sense that we want to understand all the information that the (simpler) matricizations provide about the orginial (more complicated) tensor. 

![Imgur](https://i.imgur.com/Xrk24su.jpg)
"""

# ╔═╡ ddbe64d0-0d0c-42db-99dc-50b857bea319
md"""
## The main result and some applications
"""

# ╔═╡ b8f23b1a-a70a-4209-aef3-1a99483831e3
md"""
### Warm-up: one flattening
"""

# ╔═╡ f47a4cc6-d6f8-4c1e-a954-94dd1f9a3651
md"""
Instead of going for the joint distribution of the set of all flattenings, let us consider a single flattening, corresponding to a permutation $\sigma \in \mathfrak S_{2k}$. Clearly, the choice of the permutaiton $\sigma$ is not important: 

**Fact.** The flattenings $M_\sigma$ have the same distribution. 

This follows easily from the fact that the entries of $M$ are i.i.d. hence tensor-permuting them with $\sigma$ does not change the distribution. 
"""

# ╔═╡ b985cd94-ea23-4820-b113-9bce6e4a40b3
md"""
Let us take $\sigma = \mathrm{id}$, and consider the flattening 

$X := M_{\textrm{id}} \in \mathcal M_{N^k}(\mathbb C).$
This random matrix has i.i.d. entries with a well-behaved distribution. Since we are interested in the single matrix $X$, the fact that it acts on a tensor product space $(\mathbb C^N)^{\otimes k}$ is not relevant anymore, so let us set $D = N^k$. 

Below we simulate normalized complex $D \times D$ Gaussian matrices, aka _Ginibre_ random matrices. 
"""

# ╔═╡ a064f844-eb97-4cae-84a1-2bbefa934cea
md"""
D = $@bind D Slider(10:10:1000, default=400, show_value=true)
"""

# ╔═╡ 68517295-5a72-4f1a-a376-2755084fc043
	X = randn(ComplexF64, (D, D))/sqrt(D)

# ╔═╡ 195035ae-c779-4b93-a375-100c7e9e1fc3
md"""
Standard results from random matrix theory apply, for example: 

**Theorem.** The empirical eigenvalue distribution of $X$

$\mu(X) := \frac 1 D \sum_{i=1}^D \delta_{\lambda_i(X)}$
converges, as $D \to \infty$, to the _uniform measure on the unit disk_. 
"""

# ╔═╡ a201ea32-0891-4d98-b0a7-b36e3b6dd9d7
begin
	ev = eigvals(X)
	
	plot(t->cos(t), t->sin(t), 0, 2π, fill=(0, 0.3, :lightblue), lw=3, label="Unit disk", aspect_ratio=:equal)
	plot!(real(ev), imag(ev), label="Eigenvalues of X", seriestype=:scatter, color=:gray,legend=:outerright)
		xlims!(-1.2, 1.2)
		ylims!(-1.2, 1.2)
end

# ╔═╡ 7eadd649-c1ea-4e6e-92d9-8b489d85e260
md"""
**Theorem.** The empirical squared singular value distribution of $X$

$\nu(X) := \frac 1 D \sum_{i=1}^D \delta_{s^2_i(X)}$
converges, as $D \to \infty$, to the _Marchenko-Pastur distribution_:

$\mathrm{d} \mathrm{MP}(x) := \frac{\sqrt{x(4-x)}}{2 \pi x} \mathbf 1_{(0,4)}(x) \mathrm{d}x.$
"""

# ╔═╡ 64004cc0-96ba-4df7-9f44-5f3b788aadea
begin
	histogram(eigvals(X'*X), label="Squared singular values of X", bins=range(-0.2, 4.2, length=51), normalize=:pdf, color=:gray)

	MarchenkoPastur(x) = sqrt(x*(4 - x))/(2*π*x)
	plot!(MarchenkoPastur, label="Marchenko-Pastur distribution", lw=3, color=:red,legend=:topright)
	xlims!(-0.2, 4.2)
	ylims!(0, 2.5)
end

# ╔═╡ af1beded-090c-4758-8aee-70ef3ed7c92a
md"""
In conclusion, any of the $(2k)!$ flattenings of a random tensor behaves like a _Ginibre_ random matrix, a well-studied ensemble in random matrix theory. In the setting of free probability theory, such a sequence of random matrices $X_D$ converges, as $D \to \infty$, to a _circular element_ $m$. Briefly, standard circular elements are characterized by the fact that their only non-vanishing cumulants are 
$\kappa_2(m, m^*) = 1$. Concretely, this means that we have a moment formula:

$\begin{align}
\lim_{D \to \infty} &\frac 1 D \mathbb E \operatorname{Tr} \big( X_D^{w(1)} X_D^{w(2)} \cdots X_D^{w(p)} \big) = \varphi( m^{w(1)} m^{w(2)} \cdots m^{w(p)})= \\
& |\{\text{noncrossing pairings of } [p] \text{ such that in each pair } (i,j), \, w(i) \neq w(j)\}|;
\end{align}$
above, $w$ is a word in the letters $\{1, *\}$. For example, 

$\varphi\big( (m^*m)^p \big) = |\{\text{noncrossing pairings of } [2p]\}| = \mathrm{Cat}_p = \int t^p \mathrm{d} \mathrm{MP}(t).$

Our work describes the _joint_ distribution of all the flattenings. Importantly, note that the different flattenings $M_\sigma$ are _not independent_, since they all come from the same tensor $M$. 
"""

# ╔═╡ e99e928f-f0b6-4c1a-a6da-fe960c023dbf
md"""
### Statement of the main theorem
"""

# ╔═╡ fa56ce16-d525-4c3d-bad4-40a2fd9326fd
md"""
**Theorem.** Let $M_N$ be a random tensor satisfying _Tensor-Distribution-Hypothesis_ with parameter $(c,c')$. Then the collection of flattenings of $M_N$ converges in $\mathfrak{S}_k^*$-distribution to a  centered $\mathfrak{S}_k$-circular family $\mathbf m = (m_\sigma)_{\sigma \in \mathfrak S_{2k}}$, in some space $(\mathcal A, \mathbb C \mathfrak S_k, \mathcal E)$.

...

The $\mathfrak{S}_k$-covariance of  $\mathbf m$ is given by the following equalities: 
 
...
"""

# ╔═╡ 1aced8c5-962c-401b-b6b5-6e3997f77884
md"""
### Application: symmetric tensors
"""

# ╔═╡ c5241dc3-37d4-4cfd-a049-2592102a99d3
md"""
Consider a symmetric tensor $S \in \vee^{2k}(\mathbb C^N)$; such a tensor is invariant by coordinate permutation: for all $\sigma \in \mathfrak S_{2k}$, $M = \sigma \cdot M$. 

In coordinates:

$M_{i_1  \cdots i_{2k}} = M_{i_{\sigma(1)} \cdots i_{\sigma(2k)}} \qquad \forall \sigma \in \mathfrak S_{2k}, \quad \forall i_1, \ldots, i_{2k} \in [N].$
"""

# ╔═╡ cd468e7a-b644-44dc-a487-2cd283d7b17a
md"""
We can obtain such a random symmetric tensor by projecting a random tensor $M$ onto the symmetric subspace:

$S := \frac{1}{(2k)!} \sum_{\sigma \in \mathfrak S_{2k}} \sigma \cdot M.$
"""

# ╔═╡ b02d7258-608e-4eab-b41d-addd7a2c5b9c
md"""
Since $S$ is symmetric, all its flattenings are identical: for all $\pi \in \mathfrak S_{2k}$, 

$S_\pi = \frac{1}{(2k)!}\sum_{\sigma \in \mathfrak S_{2k}} M_\sigma.$
"""

# ╔═╡ 2735d8d4-6ae3-4df0-8c92-6ec361c564e3
md"""
**Corollary.** The empirical squared  singular value distribution of the flattening of the random symmetric tensor 

$\sqrt{\frac{(2k)!}{k! c}}S$
converges, as $N \to \infty$, to a dilution of the Marchenko-Pastur distribution

$(1-1/k!) \delta_0 + (1/k!) \mathrm{MP}.$
"""

# ╔═╡ a3a4ebc2-49ed-41b1-8c42-0fbdf97622f2
md"""
Similar results hold for similar symmetric objects, such as 

$A = \sum_{\sigma \in \mathfrak S_{2k}} \operatorname{sign}(\sigma) M_\sigma$
or

$H = \sum_{\sigma \in \mathfrak S_{2k}}  M_\sigma + M^*_\sigma.$

"""

# ╔═╡ 30113942-39bc-4cc1-82b4-83b88c3fab44
md"""
## The sum of two random projections
"""

# ╔═╡ 1c20ffff-ea34-4b71-9829-570d5ab1cd15
md"""
In order to define the limits of the family of flattenings of a random tensor, we have to understand the basics of _free probability theory_ and its generalization, _operator-valued_ free probability theory, and their connection to random matrix theory. 

To do this, we shall first explore [Voiculescu](https://en.wikipedia.org/wiki/Dan-Virgil_Voiculescu)'s result about the _asymptotic freeness_ of independent, unitarily invariant random matrices, and how our model of random tensor flattenings departs from it. 
"""

# ╔═╡ e47e9884-78ef-49de-8a5b-02778fdf627e
md"""
To illustrate the relation between free probability theory and random matrix theory, we shall consider a running example consisting of two $N \times N$ orthogonal projectors, $P, Q \in \mathcal M_N(\mathbb C)$ of ranks $N/2$. 
"""

# ╔═╡ 10a86e16-d990-46cc-8966-b6f892b33d77
md"""
### Diagonal projections
"""

# ╔═╡ d359bebb-c776-411e-9c38-3a40093e1114
md"""
Let us start by sampling two such random **diagonal** projections. More precisely, we shall define diagonal matrices $P_{\text{diag}}$, $Q_{\text{diag}}$, having random 0,1 entries with probability 1/2.
"""

# ╔═╡ 8524cc4b-fa86-4af2-a131-be81bb4245d7
md"""
Let us plot the histogram of the eigenvalues of $P_{\textrm{diag}}$:
"""

# ╔═╡ 284fbd6c-1c33-4c0f-9503-eff2e1f6e2f0
md"""
In the large $N$ limit, half of the eigenvalues are 0, and the other half are +1. We say that the limiting **empirical eigenvalue distribution** of $P_{\textrm{diag}}$ is a Bernoulli distribution of parameter 1/2: 

$\lim_{N \to \infty} \sum_{i=1}^N \delta_{\lambda_i(P_{\textrm{diag}})} = \frac 1 2 \delta_0 + \frac 1 2 \delta_1.$

"""

# ╔═╡ 6d8b9010-0b60-4550-a028-3faaa266b27c
md"""
Let us now consider the eigenvalues of the **sum** $P_{\textrm{diag}} + Q_{\textrm{diag}}$:
"""

# ╔═╡ b925f359-f7e3-4a8c-a3fd-00e8981ca3ec
md"""
Clearly, the eigenvalues of sum of the two **diagonal** projections converge to the probaiblity measure 

$\frac 1 4 \delta_0 + \frac 1 2 \delta_1 + \frac 1 4 \delta_2.$

This measure is the **classical convolution** of two Bernoulli distributions with parameter 1/2:

$\big(\frac 1 2 \delta_0 + \frac 1 2 \delta_1\big) \star \big(\frac 1 2 \delta_0 + \frac 1 2 \delta_1\big) = \frac 1 4 \delta_0 + \frac 1 2 \delta_1 + \frac 1 4 \delta_2.$
"""

# ╔═╡ 073fab26-9b05-4974-8abb-14b16c4a6262
md"""
### Randomly roatated projections
"""

# ╔═╡ 345e2a92-ce28-41a2-90fa-cbf560b597ac
md"""
Let us consider now the same situation as above, but with projections on subspaces in **general position**. To achieve this, we shall randomly change bases for the diagonal projections considered previously: 

$P = U P_{\textrm{diag}} U^* \qquad Q = V Q_{\textrm{diag}} V^*,$

for _Haar-distributed_, independent random unitary matrices $U,V \in \mathcal U(N)$.
"""

# ╔═╡ 0a5e3c6f-1862-45b1-a179-f031944e67c8
md"""
N = $@bind N Slider(10:10:1000, default=400, show_value=true)
"""

# ╔═╡ ffd8e6d3-5938-41a0-9c6c-eb049a9d08d3
begin
	Pdiag = Diagonal(rand((0,1), (N,N)))
	Qdiag = Diagonal(rand((0,1), (N,N)))
end

# ╔═╡ f47bdbf7-f10e-4eff-b810-a388852c6ffa
histogram(eigvals(Pdiag), label="Eigenvalues of Pdiag", bins=range(-0.5, 1.5, length=11), normalize=:probability, color=:gray, legend=:outertop)

# ╔═╡ cbfed8d0-b657-44c6-bccc-c7e20196f3aa
histogram(eigvals(Pdiag+Qdiag), label="Eigenvalues of Pdiag+Qdiag", bins=range(-0.5, 2.5, length=16), normalize=:probability, color=:gray, legend=:topright)

# ╔═╡ 437698ab-e68c-4bac-be5f-7d041333da3f
begin
	CUEdistribution = Haar(2)
	U = rand(CUEdistribution, N)
	V = rand(CUEdistribution, N)
	
	P = U * Pdiag * U'
	Q = V * Qdiag * V'

	P = (P+P')/2
	Q = (Q+Q')/2
end

# ╔═╡ b1154d49-cbeb-4971-9154-ed548aaf6700
begin
	histogram(eigvals(P+Q), label="Eigenvalues of P+Q", bins=range(-0.2, 2.2, length=51), normalize=:pdf, color=:gray, legend=:top)
	
	arcsine(x) = 1/(π * sqrt(x*(2 - x)))
	plot!(arcsine, label="Arcsine distribution", lw=3, color=:red)
	xlims!(-0.2, 2.2)
	ylims!(0, 1.5)
end

# ╔═╡ 1b92bf08-a6ef-4f71-9152-e7822bfead25
md"""
We notice that the eigenvalues of $P+Q$ can take any value in the range $(0,2)$, and that the limiting eigenvalue distribution is given by the **arcsine distribution**:

$\lim_{N \to \infty} \sum_{i=1}^N \delta_{\lambda_i(P+Q)} =\frac{1}{\pi \sqrt{x(2-x)}} \mathbf{1}_{(0,2)}(x) \mathrm{d}x.$

The arcsine law is the **free additive convolution** of the two Bernoulli distributions: 

$\big(\frac 1 2 \delta_0 + \frac 1 2 \delta_1\big) \boxplus \big(\frac 1 2 \delta_0 + \frac 1 2 \delta_1\big) = \frac{1}{\pi \sqrt{x(2-x)}} \mathbf{1}_{(0,2)}(x) \mathrm{d}x.$
"""

# ╔═╡ 03ff5489-07bf-4354-b18a-580566509ecd
md"""
## Asymptotic freeness
"""

# ╔═╡ c9286f27-6dfb-473e-a694-a1382b1512e0
md"""
The above result is a standard exercise in **free probability theory**, and is a consequence of the following theorem of Voiculescu. 
"""

# ╔═╡ d38df688-928c-445a-a448-c1fa33f2c28a
md"""
**Theorem.** Independent, unitarily invariant random matrices are asymptotically free. 
"""

# ╔═╡ 308996e2-44b4-4330-90ca-0af518a77e60
md"""
This theorem applies directly to our random matrices $P$ and $Q$, in the large $N$ limit. Indeed: 
- the random matrices $P$ and $Q$ are independent
- the distributions of $P$ and $Q$ are unitarily invariant, since they rotated by independent Haar-distributed random unitary matrices $U,V$

We consider next what happens when we drop one of the two assumptions in Voiculescu's theorem.
"""

# ╔═╡ 9f0d0c41-fb99-49ed-8d56-5e757a2b416e
md"""
### Dropping independence 
"""

# ╔═╡ 3785e15c-6dff-42bb-8f4e-8f858bc40305
md"""
We consider now the sum of $P$ and its transpose, $P+P^\top$. Clearly, the two matrices $P$, $P^\top$ are **not independent**. 
"""

# ╔═╡ 73f81b02-0a6e-4f52-8e38-7a4a78c46d82
begin
	histogram(eigvals(P+transpose(P)), label="Eigenvalues of P+Pᵀ", bins=range(-0.2, 2.2, length=51), normalize=:pdf, color=:gray, legend=:top)
	
	plot!(arcsine, label="Arcsine distribution", lw=3, color=:red)
	xlims!(-0.2, 2.2)
	ylims!(0, 1.5)
end

# ╔═╡ d10e1f63-b49b-48e1-bc8b-2a5af8b57193
md"""
Remarkably, Mingo and Popa have shown the following result. 

**Theorem.** A unitarily invariant random matrix is asymptotically free from its transpose. 

Hence, the limiting eigenvalue distribution of $P + P^\top$ is still the arcsine law, as if the matrices were independent. 
"""

# ╔═╡ 329ac4c2-3ca1-41fc-b347-bf6deca2e2d2
md"""
We recover a special case of this result as a corollary of our main theorem, in the case $k=2$, since 

$M_{(12)} = \big( M_{(1)(2)} \big)^\top,$

and the covariance of the two permutations in $\mathfrak S_2$ is zero. In other words, the two matrices $M_{(12)}$ and $M_{(1)(2)}$ are asymptotically *free circular elements*; they behave as if they were _independentent_ Ginibre matrices.  
"""

# ╔═╡ 1f5066f6-c8d4-4b92-9fa0-f651f1d991dc
md"""
This apparent independence appears for all values of $k$, and can be readily read out from the (operator-valued) covariance of the circular elements in our main theorem. 

To explain this, we need to introduce some group-theoretic machinery. Recall that to a pair permutations $\eta, \eta' \in \mathfrak S_k$, we associate the permutation $\eta \sqcup \eta' \in \mathfrak S_{2k}$ which acts like $\eta$ on the first $k$ elements and as $\eta'$ on the last $k$ elements of $[2k]$. Such permutations $\eta \sqcup \eta'$ form a subgroup $\mathfrak S_{k,k} \leq \mathfrak S_{2k}$.

Two flattenings $M_\sigma$ and $M_{\eta \sqcup \eta' \cdot \sigma}$ are related by _deterministic_ tensor-permutation matrices

$M_{\eta \sqcup \eta' \cdot \sigma} = \eta \cdot M_{\sigma} \cdot (\eta')^*,$
so they have no chance of being asymptotically free. As it turns out, this is (morally) the only obstruction: 

**Corollary.** A family of flattenings $\{M_\sigma\}$ with the propery that the permutations $\sigma$ belong to _disjoint_ $\mathfrak S_{k,k} \backslash \mathfrak S_{2k}$ _cosets_ is asymptotically free. 

Note that there are precisely $\binom{2k}{k}$ such cosets; they correpond to choosing a k-subset of the $2k$ legs which will be output legs, with the complement being input legs. 
"""

# ╔═╡ 8f2b6a44-88ce-4bd5-bcfe-0d9b84ccbb07
md"""
### Dropping unitary invariance. Block matrices
"""

# ╔═╡ dcb61f30-3742-4fc1-bd8b-18c7cb50beaf
md"""
We shall now consider the following two block matrices: 

$P_{\textrm{block}} := \begin{bmatrix}
	P & 0 \\ 
	0 & P
\end{bmatrix} = P \otimes I_2 \qquad \text{and} \qquad Q_{\textrm{block}} := \begin{bmatrix}
	0 & Q \\ 
	Q & 0
\end{bmatrix} = Q \otimes \begin{bmatrix}
	0 & 1 \\ 
	1 & 0
\end{bmatrix}.$

Clearly, they are independent, but **not unitarily invariant** (although $P$ and $Q$ are). 

The limiting eigenvalue distributions of these matrices are, respectively, 

$\frac 1 2 \delta_0 + \frac 1 2 \delta_1 \qquad \text{and} \qquad \frac 1 4 \delta_{-1} + \frac 1 2 \delta_0 + \frac 1 4 \delta_1.$

We are interested in their sum $P_{\textrm{block}}+Q_{\textrm{block}}$, in the large $N$ limit.
"""

# ╔═╡ 40487aae-8fb6-45c8-aef0-4cbdb5ae0905
begin
	Pblock = kron(P, [1 0; 0 1])
	Qblock = kron(Q, [0 1; 1 0])

	histogram(eigvals(Pblock + Qblock), label="Eigenvalues of Pblock+Qblock", bins=range(-1.2, 2.2, length=51), normalize=:pdf, color=:gray, legend=:outertop)
end

# ╔═╡ e5260be3-96ff-4ea8-8b2d-cc7e2815edfd
md"""
Note that if we rotate the second matrix $Q_{\textrm{block}}$ by a random unitary matrix of full size ($2N$):

${Q_{\textrm{block}}^{\textrm{UI}}} := W Q_{\textrm{block}} W^* \qquad \text{ for } W \in \mathcal U(2N),$

we obtain a different eigenvalue distribution for $P_{\textrm{block}} + {Q_{\textrm{block}}^{\textrm{UI}}}$:
"""

# ╔═╡ 84d0c7f4-7ede-4671-aa5c-17aa8e824065
begin
	W = rand(CUEdistribution, 2*N)
	QblockUI = W * Qblock * W'
	QblockUI = (QblockUI + QblockUI')/2

	histogram(eigvals(Pblock + QblockUI), label="Eigenvalues of Pblock+Qblock", bins=range(-1.2, 2.2, length=51), normalize=:pdf, color=:gray, legend=:outertop)
end

# ╔═╡ 62125591-5410-48ad-bc5e-17910b87876d
md"""
The last two plots are visually different, the limiting distributions are not identical. This can also be seen by looking at the following mixed moment which differ in the two situations:

$\frac{1}{2N} \operatorname{Tr}\big[ P_{\textrm{block}} Q^\epsilon_{\textrm{block}} P_{\textrm{block}} Q^\epsilon_{\textrm{block}} ] \qquad \text{for } \epsilon \in \{\emptyset, \textrm{UI}\}.$
"""

# ╔═╡ f0d83c87-39d0-447e-ac82-35b27a950d23
(1/(2*N))tr(Pblock*Qblock*Pblock*Qblock)

# ╔═╡ fa6f940b-1e0c-49ca-a959-39c65e5f8758
(1/(2*N))tr(Pblock*QblockUI*Pblock*QblockUI)

# ╔═╡ f9af5ce9-678b-4c80-9c88-99589ad917e3
md"""
In the first situation, the random matrices $P_{\textrm{block}}$ and $Q_{\textrm{block}}$ are not unitarily invariant, and one cannot apply Voiculescu's theorem. As it turns out, they are not asymptotically free, but only _operator-valued asymptotically free_; more precisely, they are $\mathcal M_2(\mathbb C)$-free.

In the second situation, the random matrices $P_{\textrm{block}}$ and $Q^{\textrm{UI}}_{\textrm{block}}$ satisfy the assumptions of Voiculescu's theorem, so the limiting probability distribution is a _free additive convolution_:

$\big(\frac 1 2 \delta_0 + \frac 1 2 \delta_1 \big) \boxplus \big( \frac 1 4 \delta_{-1} + \frac 1 2 \delta_0 + \frac 1 4 \delta_1\big).$
"""

# ╔═╡ 78326f4e-3cca-403c-831e-bbd0501ddf6f
md"""
Going back to tensor flattenings, as mentioned previously, flattenings related by permutations of the form $\eta \sqcup \eta'$ are related by deterministic tensor-permutation matrices. This is the reason why we need to take into accound these matrices, which in the limit converge (in distribution) to the group algebra $\mathbb C \mathfrak S_k$. The limiting elements $m_\sigma$ satisfy the same relations
$m_{\eta \sqcup \eta' \cdot \sigma} = \eta \cdot m_{\sigma} \cdot (\eta')^*.$
""" 

# ╔═╡ 298fcd42-5d29-42df-a7b1-29882ff6227d
md"""
## Future directions
"""

# ╔═╡ a9995743-33b6-4b59-9a3d-0a35abda700a
md"""
In this work, we have considered the joint distribution of a set of matrices (tensor flattenings). This allows us to compute limits of expectations of traces of words in these matrices. 

However, one would like to compute expectations of more complicated _tensor diagrams_ of random tensors, in a systematic way. For example, this would allow us to compute the limit eigenvalue distribution of the _partial transposition_ of a (symmetric) tensor flattening. 

New tools are needed for this, since exotic behavior might occur, such as outlier eigenvalues, see joint work with Stéphane Dartois and Adrian Tanasa [arXiv:2111.05638](https://arxiv.org/abs/2111.05638).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RandomMatrices = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "440eebf9025f406f47c368e7c6b9e2cc88513f68"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "93ff6a4d5e7bfe27732259bfabbdd19940d8af1f"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.GSL]]
deps = ["GSL_jll", "Libdl", "Markdown"]
git-tree-sha1 = "3ebd07d519f5ec318d5bc1b4971e2472e14bd1f0"
uuid = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
version = "1.0.1"

[[deps.GSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "56f1e2c9e083e0bb7cf9a7055c280beb08a924c0"
uuid = "1b77fbbe-d8ee-58f0-85f9-836ddc23a7a4"
version = "2.7.2+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "66b2fcd977db5329aa35cac121e5b94dd6472198"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.28"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomMatrices]]
deps = ["Combinatorics", "Distributions", "FastGaussQuadrature", "GSL", "LinearAlgebra", "Random", "SpecialFunctions", "Test"]
git-tree-sha1 = "a2218db37fb243b0f3808904304a60b559f4e2c0"
uuid = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
version = "0.5.3"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "49cbf7c74fafaed4c529d47d48c8f7da6a19eb75"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.1"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─04312c66-7d48-11ee-1562-83bf7dd613d8
# ╟─43b09095-6198-4544-8137-00fab6f5ff11
# ╟─3a4824db-9674-45c1-90e4-c7a0c9cac464
# ╟─58903216-1306-4fc1-8e25-658c4a3b7435
# ╟─b0b0a74b-f196-402b-a9d2-45758053542a
# ╟─bcdde9e3-aebd-4795-80db-af957a15764a
# ╟─c5bda79a-c3df-45ad-bc6f-9e61dd155cff
# ╟─874575ff-0390-4bd4-a2dd-f9205e0c83ab
# ╟─66e0d010-1508-46d4-bd24-0e203095342f
# ╟─b75ea572-c70b-4f36-836b-941350fb099f
# ╟─37aee748-f6bc-4421-9c37-e163308d71b9
# ╟─b8f00bcb-0cf9-4795-9405-2d6381f0b4ff
# ╟─88fa8bb5-f7c4-4215-8430-666a899d4ad9
# ╟─d3f708a7-4a0e-4779-bf93-16d2494f5976
# ╟─59ce6289-e929-4fdd-9bd3-447966b55691
# ╟─530fa17a-61cf-4799-aac5-2ee82dfab4b5
# ╟─ddbe64d0-0d0c-42db-99dc-50b857bea319
# ╟─b8f23b1a-a70a-4209-aef3-1a99483831e3
# ╟─f47a4cc6-d6f8-4c1e-a954-94dd1f9a3651
# ╟─b985cd94-ea23-4820-b113-9bce6e4a40b3
# ╠═a064f844-eb97-4cae-84a1-2bbefa934cea
# ╠═68517295-5a72-4f1a-a376-2755084fc043
# ╟─195035ae-c779-4b93-a375-100c7e9e1fc3
# ╠═a201ea32-0891-4d98-b0a7-b36e3b6dd9d7
# ╟─7eadd649-c1ea-4e6e-92d9-8b489d85e260
# ╠═64004cc0-96ba-4df7-9f44-5f3b788aadea
# ╟─af1beded-090c-4758-8aee-70ef3ed7c92a
# ╟─e99e928f-f0b6-4c1a-a6da-fe960c023dbf
# ╟─fa56ce16-d525-4c3d-bad4-40a2fd9326fd
# ╟─1aced8c5-962c-401b-b6b5-6e3997f77884
# ╟─c5241dc3-37d4-4cfd-a049-2592102a99d3
# ╟─cd468e7a-b644-44dc-a487-2cd283d7b17a
# ╟─b02d7258-608e-4eab-b41d-addd7a2c5b9c
# ╟─2735d8d4-6ae3-4df0-8c92-6ec361c564e3
# ╟─a3a4ebc2-49ed-41b1-8c42-0fbdf97622f2
# ╟─30113942-39bc-4cc1-82b4-83b88c3fab44
# ╟─1c20ffff-ea34-4b71-9829-570d5ab1cd15
# ╟─e47e9884-78ef-49de-8a5b-02778fdf627e
# ╟─10a86e16-d990-46cc-8966-b6f892b33d77
# ╟─d359bebb-c776-411e-9c38-3a40093e1114
# ╠═ffd8e6d3-5938-41a0-9c6c-eb049a9d08d3
# ╟─8524cc4b-fa86-4af2-a131-be81bb4245d7
# ╠═f47bdbf7-f10e-4eff-b810-a388852c6ffa
# ╟─284fbd6c-1c33-4c0f-9503-eff2e1f6e2f0
# ╟─6d8b9010-0b60-4550-a028-3faaa266b27c
# ╠═cbfed8d0-b657-44c6-bccc-c7e20196f3aa
# ╟─b925f359-f7e3-4a8c-a3fd-00e8981ca3ec
# ╟─073fab26-9b05-4974-8abb-14b16c4a6262
# ╟─345e2a92-ce28-41a2-90fa-cbf560b597ac
# ╠═0a5e3c6f-1862-45b1-a179-f031944e67c8
# ╠═437698ab-e68c-4bac-be5f-7d041333da3f
# ╠═b1154d49-cbeb-4971-9154-ed548aaf6700
# ╟─1b92bf08-a6ef-4f71-9152-e7822bfead25
# ╟─03ff5489-07bf-4354-b18a-580566509ecd
# ╟─c9286f27-6dfb-473e-a694-a1382b1512e0
# ╟─d38df688-928c-445a-a448-c1fa33f2c28a
# ╟─308996e2-44b4-4330-90ca-0af518a77e60
# ╟─9f0d0c41-fb99-49ed-8d56-5e757a2b416e
# ╟─3785e15c-6dff-42bb-8f4e-8f858bc40305
# ╠═73f81b02-0a6e-4f52-8e38-7a4a78c46d82
# ╟─d10e1f63-b49b-48e1-bc8b-2a5af8b57193
# ╟─329ac4c2-3ca1-41fc-b347-bf6deca2e2d2
# ╟─1f5066f6-c8d4-4b92-9fa0-f651f1d991dc
# ╟─8f2b6a44-88ce-4bd5-bcfe-0d9b84ccbb07
# ╟─dcb61f30-3742-4fc1-bd8b-18c7cb50beaf
# ╠═40487aae-8fb6-45c8-aef0-4cbdb5ae0905
# ╟─e5260be3-96ff-4ea8-8b2d-cc7e2815edfd
# ╠═84d0c7f4-7ede-4671-aa5c-17aa8e824065
# ╟─62125591-5410-48ad-bc5e-17910b87876d
# ╠═f0d83c87-39d0-447e-ac82-35b27a950d23
# ╠═fa6f940b-1e0c-49ca-a959-39c65e5f8758
# ╟─f9af5ce9-678b-4c80-9c88-99589ad917e3
# ╟─78326f4e-3cca-403c-831e-bbd0501ddf6f
# ╟─298fcd42-5d29-42df-a7b1-29882ff6227d
# ╟─a9995743-33b6-4b59-9a3d-0a35abda700a
# ╟─03b1e4ea-d13f-4f2f-adc6-341598a30658
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
