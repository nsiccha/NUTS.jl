using DynamicHMC, NUTS, Random, Distributions, LinearAlgebra, LogExpFunctions, Chairmarks, LogDensityProblems

merge_expr(x) = x
merge_expr(x::Expr) = if x.head == :(=) && Meta.isexpr(x.args[1], :(.))
    obj, name = x.args[1].args
    rhs = (x.args[2])
    merge_expr(:($obj = merge($obj, (;$name=>$rhs))))
else
    x
end
__!!__expr(x) = x
__!!__expr(x::Expr) = if x.head == :call
    if endswith(string(x.args[1]), "!!")
        merge_expr(Expr(:(=), x.args[2], x))
    else
        x
    end
elseif x.head == :(=) && Meta.isexpr(x.args[1], :(.))
    merge_expr(x)
elseif x.head == :macrocall
    x
else
    Expr(x.head, __!!__expr.(x.args)...)
end
macro __!!__(x)
    esc(__!!__expr(x))
end
init!!_expr(x) = x
init!!_expr(x::Expr) = if x.head == :(=) && Meta.isexpr(x.args[1], :(.))
    obj, name = x.args[1].args
    rhs = x.args[2]
    :(hasproperty($obj, $name) || (println("Initializing ", $name); $obj = merge($obj, (;$name=>$rhs))))
else
    Expr(x.head, init!!_expr.(x.args)...) 
end
macro init!!(x)
    esc(init!!_expr(x))
end

begin 

    leapfrog!!(state, cfg) = @__!!__ begin 
        @. state.momentum += .5 * cfg.stepsize * state.log_density_gradient 
        state.velocity .= state.momentum
        @. state.position += cfg.stepsize * state.velocity
        log_density_gradient!!(state, cfg.posterior)
        @. state.momentum += .5 * cfg.stepsize * state.log_density_gradient 
        state.velocity .= state.momentum
        state
    end
    log_density_gradient!!(state, posterior) = @__!!__ begin 
        state.log_density = log_density_gradient!(posterior, state.position, state.log_density_gradient)
    end
    log_density_gradient!(d::IsoNormal, x::AbstractVector, g::AbstractVector) = begin
        @. g = -x
        # logpdf(d, x)
        .5 * dot(x,x)
    end
    hamiltonian!!(state) = @__!!__ state.hamiltonian = -state.log_density + .5 * dot(state.velocity, state.momentum) 

    phase_point(position, log_density, log_density_gradient, momentum, velocity) = (;position, log_density, log_density_gradient, momentum, velocity)
    phase_point(d::Integer) = phase_point(zeros(d), 0., zeros(d), zeros(d), zeros(d))
    phase_point(state::NamedTuple) = begin
        rv = phase_point(length(state.position))
        rv.position .= state.position
        log_density_gradient!!(rv, state.posterior)
    end
    trajectory(d::Int) = trajectory(zeros(d), zeros(d))
    trajectory(bwd, fwd) = (;bwd, fwd)
    mv(momentum, velocity) = (;momentum, velocity)
    mv(d::Int) = mv(zeros(d), zeros(d))
    @inline mcopy!(x::NamedTuple, y::NamedTuple) = map(mcopy!, x, y)
    @inline @generated mcopy2!(x, y) = Expr(:block, [
        :(mcopy2!(x.$name, y.$name)) for name in fieldnames(y) if name in fieldnames(x)
    ]..., :x)
    @inline mcopy2!(x, y::AbstractArray) = copy!(x, y)
    @inline copy!!(x, y) = y
    @inline copy!!(x::AbstractArray, y::AbstractArray) = copy!(x, y)
    @inline @generated copy!!(x::T1, y::T2) where {T1,T2<:NamedTuple} = :(
        merge(x, (;$([:($(Meta.quot(name))=>copy!!(x.$name, y.$name)) for name in fieldnames(T2) if name in fieldnames(T1)]...)))
    )
    # dtrajectory(d::Int) = dtrajectory(zeros(d), zeros(d), zeros(d), zeros(d), zeros(d), zeros(d))
    # dtrajectory(bwd, bwd_bwd, bwd_fwd, fwd_bwd, fwd_fwd, fwd) = (;bwd, bwd_bwd, bwd_fwd, fwd_bwd, fwd_fwd, fwd)
    dtrajectory(d::Int) = dtrajectory(zeros(d), zeros(d), zeros(d), zeros(d))
    dtrajectory(bwd, bwd_fwd, fwd_bwd, fwd) = (;bwd, bwd_fwd, fwd_bwd, fwd)
    tree(dimension) = (;
        log_sum_weight=trajectory(-Inf,-Inf),
        bwd=mv(dimension),
        bwd_fwd=mv(dimension),
        fwd=mv(dimension),
        summed_momentum=trajectory(dimension),
    )
    trees(dimension, max_depth) = map(i->tree(dimension), 1:max_depth)
    proposal(dimension) = (;
        position=zeros(dimension),
        log_density_gradient=zeros(dimension)
    )
    proposals(dimension, max_depth) = map(i->proposal(dimension), 1:max_depth)
    compute_criterion(momentum, bwd, fwd) = (dot(momentum, bwd) > 0 && dot(momentum, fwd) > 0)
    badd(args...) = Base.broadcasted(+, args...)
    swapproposal!(state, i, j=length(state.proposals)) = begin 
        state.proposals[i], state.proposals[j] = state.proposals[j], state.proposals[i]
    end
    nuts!!(state) = @__!!__ begin 
        dimension = length(state.position)
        @init!! begin
            state.current = phase_point(state)
            state.trees = trees(dimension, state.max_depth+1)
            state.proposals = proposals(dimension, state.max_depth+2)
        end 
        randn!(state.rng, state.current.momentum)
        state.current.velocity .= state.current.momentum
        hamiltonian!!(state.current)
        state.init_hamiltonian = state.current.hamiltonian
        state.n_leapfrog = 0
        state.may_sample = true
        state.may_continue = true
        state.divergent = false
        state.sum_metro_prob = 0.
        state.trees[1].log_sum_weight.fwd = 0.
        copy!!(state.proposals[1], state.current)
        for depth in 1:state.max_depth
            nuts_finish_tree!!(state, depth)
            tree = state.trees[depth]
            state.may_sample || break
            randbernoullilog(state.rng, tree.log_sum_weight.fwd - tree.log_sum_weight.bwd) && swapproposal!(state, depth+1)
            state.may_continue || break
        end
        copy!!(state.current, state.proposals[end])
        state.position .= state.current.position
        state.accept_prob = state.sum_metro_prob / (state.n_leapfrog)
        state
    end
    randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob 
    nuts_finish_tree!!(state, depth) = @__!!__ begin
        tree = state.trees[depth].log_sum_weight.bwd = state.trees[depth].log_sum_weight.fwd
        suptree = state.trees[depth+1]
        swapproposal!(state, depth, depth+1)
        if depth == 1
            mcopy2!(tree.bwd, state.current)
            tree.summed_momentum.bwd .= state.current.momentum
        else
            tree.summed_momentum.bwd .= tree.summed_momentum.fwd
        end
        mcopy2!(suptree.bwd, tree.bwd)
        mcopy2!(suptree.bwd_fwd, state.current)
        nuts_tree!!(state, depth)
        state.may_continue || return state.may_sample = false
        suptree = state.trees[depth+1].log_sum_weight.fwd = logaddexp(tree.log_sum_weight.fwd, tree.log_sum_weight.bwd)
        randbernoullilog(state.rng, tree.log_sum_weight.fwd - suptree.log_sum_weight.fwd) && swapproposal!(state, depth, depth+1)
        suptree.summed_momentum.fwd .= tree.summed_momentum.bwd .+ tree.summed_momentum.fwd
        state.may_continue = if depth == 1
            compute_criterion(suptree.summed_momentum.fwd, tree.bwd.velocity, state.current.velocity)
        else
            subtree = state.trees[depth-1]
            (
                compute_criterion(suptree.summed_momentum.fwd, tree.bwd.velocity, state.current.velocity) && 
                compute_criterion(badd(tree.summed_momentum.bwd, subtree.bwd.momentum), tree.bwd.velocity, subtree.bwd.velocity) &&
                compute_criterion(badd(tree.bwd_fwd.momentum, tree.summed_momentum.fwd), tree.bwd_fwd.velocity, state.current.velocity) 
            )
        end
    end
    finiteorinf(x) = isfinite(x) ? x : typeof(x)(Inf)
    min1exp(x) = x >= 0 ? one(x) : exp(x)
    nuts_tree!!(state, depth) = @__!!__ if depth == 1
        leapfrog!!(state.current, state)
        hamiltonian!!(state.current)
        dhamiltonian = finiteorinf(state.current.hamiltonian - state.init_hamiltonian)
        state.n_leapfrog = state.n_leapfrog + 1
        state.may_continue = dhamiltonian <= state.max_dhamiltonian
        state.divergent = !state.may_continue
        state.sum_metro_prob = state.sum_metro_prob + min1exp(dhamiltonian)
        state.trees[1].log_sum_weight.fwd = dhamiltonian
        mcopy2!(state.proposals[1], state.current)
        state
    else
        nuts_tree!!(state, depth-1)
        state.may_continue || return state
        nuts_finish_tree!!(state, depth-1)
    end
    d = 100
    posterior = MultivariateNormal(zeros(d), I)
    stepsize = .001
    position = randn(d)
    seconds = d * 10/10_000
    state = nuts!!(
        (;
            rng=Xoshiro(0),
            posterior,
            stepsize,
            max_depth=10,
            max_dhamiltonian=1000,
            position,
        )
    )
    @error (;state.n_leapfrog, state.may_sample, state.may_continue, state.divergent)
    # nothing
    display(@be nuts!!(state) seconds=seconds)
# end
# begin 
    LogDensityProblems.capabilities(::IsoNormal) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(d::IsoNormal) = length(mean(d))
    LogDensityProblems.logdensity_and_gradient(d::IsoNormal, x::AbstractVector) = .5 * dot(x,x), -x
    rng = Xoshiro(0)
    algorithm = DynamicHMC.NUTS()
    H = DynamicHMC.Hamiltonian(DynamicHMC.GaussianKineticEnergy(Diagonal(ones(d))), posterior)
    Q = DynamicHMC.evaluate_ℓ(posterior, position; strict=true)
    Q, stats = DynamicHMC.sample_tree(rng, algorithm, H, Q, stepsize)
    display(stats)
    display(@be DynamicHMC.sample_tree(rng, algorithm, H, Q, stepsize) seconds=seconds)
end