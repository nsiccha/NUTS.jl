module NUTS

export nuts!!

"""
Implements the No-U-Turn Sampler with multinomial sampling and the strict generalized no-u-turn critertion (CURRENTLY WITH IDENTITY METRIC).

Takes a single `state` argument and returns `new_state`, which contains the new sample's position at `new_state.position` / `new_state.current.position` and the new sample's gradient at `new_state.current.gradient`.

The single `state` argument should behave like a NamedTuple with required properties

    * rng,
    * posterior (implementing `NUTS.log_density_gradient!`),
    * stepsize,
    * position,

and optional properties

    * max_depth = 10,
    * min_dhamiltonian = -1000. (divergence threshold),
    * current (current/initial phase point, i.e. the position, log_density, log_density_gradient, momentum, velocity, and hamiltonian),
    * scale (TO BE IMPLEMENTED),
    * first (internal, leftmost phase point),
    * trees (internal, memory needed for recursion),
    * proposals (internal, memory needed for recursion).

After first use, optional properties will be set in the returned `new_state` and reused in subsequent calls.
"""
function nuts!! end
"Posteriors need to implement `log_density = NUTS.log_density_gradient!(posterior, position, log_density_gradient)`,
i.e. return the log density and write its gradient into `log_density_gradient`"
function log_density_gradient! end

using LinearAlgebra, Random, LogExpFunctions

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
    # :(hasproperty($obj, $name) || (println("Initializing ", $name); $obj = merge($obj, (;$name=>$rhs))))
    :(hasproperty($obj, $name) || ($obj = merge($obj, (;$name=>$rhs))))
else
    Expr(x.head, init!!_expr.(x.args)...) 
end
macro init!!(x)
    esc(init!!_expr(x))
end

leapfrog!!(state, cfg) = @__!!__ begin 
    @. state.momentum += .5 * cfg.stepsize * state.log_density_gradient 
    mul!(state.position, cfg.scale, state.momentum, cfg.stepsize, 1.)
    log_density_gradient!!(state, cfg.posterior)
    @. state.momentum += .5 * cfg.stepsize * state.log_density_gradient 
    mul!(state.velocity, cfg.scale, state.momentum)
    state
end
log_density_gradient!!(state, posterior) = @__!!__ begin 
    state.log_density = log_density_gradient!(posterior, state.position, state.log_density_gradient)
end
hamiltonian!!(state) = @__!!__ state.hamiltonian = -state.log_density + .5 * dot(state.velocity, state.momentum) 

phase_point(position, log_density, log_density_gradient, momentum, velocity, hamiltonian) = (;position, log_density, log_density_gradient, momentum, velocity, hamiltonian)
phase_point(d::Integer) = phase_point(zeros(d), 0., zeros(d), zeros(d), zeros(d), 0.)
phase_point(state::NamedTuple) = begin
    rv = phase_point(length(state.position))
    rv.position .= state.position
    log_density_gradient!!(rv, state.posterior)
end
trajectory(d::Int) = trajectory(zeros(d), zeros(d))
trajectory(bwd, fwd) = (;bwd, fwd)
mv(momentum, velocity) = (;momentum, velocity)
mv(d::Int) = mv(zeros(d), zeros(d))
@inline copy!!(x, y) = error()
@inline copy!!(x, y::Real) = y
@inline copy!!(x::AbstractArray, y::AbstractArray) = copy!(x, y)
@inline @generated copy!!(x::T1, y::T2) where {T1,T2<:NamedTuple} = :(
    merge(x, (;$([:($(Meta.quot(name))=>copy!!(x.$name, y.$name)) for name in fieldnames(T2) if name in fieldnames(T1)]...)))
)
tree(dimension) = (;
    log_weight=trajectory(-Inf,-Inf),
    bwd=mv(dimension),
    bwd_fwd=mv(dimension),
    summed_momentum=trajectory(dimension),
)
trees(dimension, max_depth) = map(i->tree(dimension), 1:max_depth+1)
proposal(dimension) = (;
    position=zeros(dimension),
    log_density=0.,
    log_density_gradient=zeros(dimension)
)
proposals(dimension, max_depth) = map(i->proposal(dimension), 1:max_depth+2)
swapproposal!(state, i, j=length(state.proposals)) = begin 
    state.proposals[i], state.proposals[j] = state.proposals[j], state.proposals[i]
end
nuts!!(state) = @__!!__ begin 
    dimension = length(state.position)
    @init!! begin
        state.max_depth = 10
        state.min_dhamiltonian = -1000.
        state.first = phase_point(dimension)
        state.current = phase_point(state)
        state.trees = trees(dimension, state.max_depth)
        state.proposals = proposals(dimension, state.max_depth)
        state.scale = I
    end 
    randn!(state.rng, state.current.momentum)
    mul!(state.current.velocity, state.scale, state.current.momentum)
    hamiltonian!!(state.current)
    state.init_hamiltonian = state.current.hamiltonian
    copy!!(state.first, state.current)
    state.first.momentum .*= -1
    state.first.velocity .*= -1
    state.n_leapfrog = 0
    state.may_sample = true
    state.may_continue = true
    state.divergent = false
    state.sum_metro_prob = 0.
    copy!!(state.proposals[end], state.current)
    state.trees[1].log_weight.fwd = 0.
    copy!!(state.proposals[1], state.current)
    for depth in 1:state.max_depth
        nuts_finish_tree!!(state, depth, rand(state.rng, Bool))
        state.may_sample || break
        tree = state.trees[depth]
        randbernoullilog(state.rng, tree.log_weight.fwd - tree.log_weight.bwd) && swapproposal!(state, depth)
        state.may_continue || break
    end
    copy!!(state.current, state.proposals[end])
    state.position .= state.current.position
    state.accept_prob = state.sum_metro_prob / (state.n_leapfrog)
    state
end
randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob 
badd(args...) = Base.broadcasted(+, args...)
compute_criterion(momentum, bwd, fwd) = (dot(momentum, bwd) > 0 && dot(momentum, fwd) > 0)
nuts_finish_tree!!(state, depth, turn) = @__!!__ begin 
    if turn
        tmp = state.current
        state.current = state.first
        state.first = tmp
        if depth > 1
            tree = state.trees[depth]
            tree.bwd.momentum .= .-state.first.momentum
            tree.bwd.velocity .= .-state.first.velocity
            tree.summed_momentum.fwd .*= -1
        end
    end 
    nuts_finish_tree!!(state, depth)
end
nuts_finish_tree!!(state, depth) = @__!!__ begin
    tree = state.trees[depth].log_weight.bwd = state.trees[depth].log_weight.fwd
    suptree = state.trees[depth+1]
    if depth == 1
        copy!!(suptree.bwd, state.current)
    else
        copy!!(suptree.bwd, tree.bwd)
        copy!!(tree.bwd_fwd, state.current)
        tree.summed_momentum.bwd .= tree.summed_momentum.fwd
    end
    nuts_tree!!(state, depth)
    state.may_continue || return state.may_sample = false
    state.trees[depth+1].log_weight.fwd = logaddexp(state.trees[depth].log_weight.fwd, state.trees[depth].log_weight.bwd)
    state.may_continue = if depth == 1
        suptree.summed_momentum.fwd .= suptree.bwd.momentum .+ state.current.momentum
        compute_criterion(suptree.summed_momentum.fwd, suptree.bwd.velocity, state.current.velocity)
    else
        suptree.summed_momentum.fwd .= tree.summed_momentum.bwd .+ tree.summed_momentum.fwd
        (
            compute_criterion(suptree.summed_momentum.fwd, suptree.bwd.velocity, state.current.velocity) && 
            compute_criterion(badd(tree.summed_momentum.bwd, tree.bwd.momentum), suptree.bwd.velocity, tree.bwd.velocity) &&
            compute_criterion(badd(tree.bwd_fwd.momentum, tree.summed_momentum.fwd), tree.bwd_fwd.velocity, state.current.velocity)
        )
    end
end
finiteorneginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
min1exp(x) = x >= 0 ? one(x) : exp(x)
nuts_tree!!(state, depth) = @__!!__ if depth == 1
    leapfrog!!(state.current, state)
    hamiltonian!!(state.current)
    dhamiltonian = finiteorneginf(state.init_hamiltonian - state.current.hamiltonian)
    state.n_leapfrog = state.n_leapfrog + 1
    state.may_continue = dhamiltonian >= state.min_dhamiltonian
    state.divergent = !state.may_continue
    state.sum_metro_prob = state.sum_metro_prob + min1exp(dhamiltonian)
    state.trees[1].log_weight.fwd = dhamiltonian
    copy!!(state.proposals[1], state.current)
    state
else
    nuts_tree!!(state, depth-1)
    state.may_continue || return state.may_sample = false
    swapproposal!(state, depth-1, depth)
    nuts_finish_tree!!(state, depth-1)
    state.may_sample || return state
    (randbernoullilog(state.rng, state.trees[depth-1].log_weight.fwd - state.trees[depth].log_weight.fwd)) && swapproposal!(state, depth-1, depth)
    state
end


end # module NUTS
