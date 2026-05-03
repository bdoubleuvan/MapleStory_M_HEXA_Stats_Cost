
using DelimitedFiles
using PyFormattedStrings
using StatsBase

using CUDA
using CUDA: i32


function compute_hexa_reset(n::Int, probability_main::Array, cost_data::Array,
                            goal_level::Int, reset_at_attempt::Int, reset_below_level::Int,
                            n_threads::Int, simulate_main::Bool)

    probability_main = CuArray{Float32}(probability_main)
    cost_data = CuArray{Float32}(cost_data)

    main = CUDA.zeros(Int32, n)
    secondary_1 = CUDA.zeros(Int32, n)
    secondary_2 = CUDA.zeros(Int32, n)
    cost = CUDA.zeros(Int32, n)
    reset = CUDA.zeros(Float32, n)

    n_blocks = div(n, n_threads) + Int32(1) * ((n % n_threads) > 0)

    if simulate_main
        @cuda threads=n_threads blocks=n_blocks hexa_reset_main_kernel(main, secondary_1, secondary_2, cost, reset,
                                                                       probability_main, cost_data, goal_level, reset_at_attempt, reset_below_level)
    else
        @cuda threads=n_threads blocks=n_blocks hexa_reset_secondary_kernel(main, secondary_1, secondary_2, cost, reset,
                                                                            probability_main, cost_data, goal_level, reset_at_attempt, reset_below_level)
    end

    return main, secondary_1, secondary_2, cost, reset
end


function hexa_reset_main_kernel(main, secondary_1, secondary_2, cost, reset,
                                probability_main, cost_data, goal_level, reset_at_attempt, reset_below_level)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    attempt = 0i32
    
    while main[i] < goal_level
        i_data = main[i] + 1i32
        attempt += 1i32

        @inbounds cost[i] += cost_data[i_data]

        roll = rand(Float32)
        
        if roll < probability_main[i_data]
            @inbounds main[i] += 1i32
        elseif (secondary_1[i] < 10i32) && (secondary_2[i] < 10i32)
            roll = rand(Float32)
            if roll <= 0.5f0
                @inbounds secondary_1[i] += 1i32
            else
                @inbounds secondary_2[i] += 1i32
            end
        elseif secondary_1[i] >= 10i32
            @inbounds secondary_2[i] += 1i32
        else
            @inbounds secondary_1[i] += 1i32
        end

        # check when a certain number of attempts have been hit and main is below a certain level
        # check if goal is reachable
        # check if no attempts left and goal has not been reached
        reset_check = 1i32 * ((attempt == reset_at_attempt) && (main[i] < reset_below_level))
        reset_check += 1i32 * ((20i32 - attempt) < (goal_level - main[i]))
        reset_check += 1i32 * ((attempt == 20i32) && (main[i] < goal_level))

        if reset_check >= 1
            @inbounds reset[i] += 1i32
            attempt = 0i32

            @inbounds main[i] = 0i32
            @inbounds secondary_1[i] = 0i32
            @inbounds secondary_2[i] = 0i32
        end
    end
    
    return
end


function hexa_reset_secondary_kernel(main, secondary_1, secondary_2, cost, reset,
                                     probability_main, cost_data, goal_level, reset_at_attempt, reset_below_level)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    attempt = 0i32
    
    while (main[i] < 8) && (secondary_1[i] < goal_level) && (secondary_2[i] < goal_level)
        i_data = main[i] + 1i32
        attempt += 1i32

        @inbounds cost[i] += cost_data[i_data]

        roll = rand(Float32)
        
        if roll < probability_main[i_data]
            @inbounds main[i] += 1i32
        elseif (secondary_1[i] < 10i32) && (secondary_2[i] < 10i32)
            roll = rand(Float32)
            if roll <= 0.5f0
                @inbounds secondary_1[i] += 1i32
            else
                @inbounds secondary_2[i] += 1i32
            end
        elseif secondary_1[i] >= 10i32
            @inbounds secondary_2[i] += 1i32
        else
            @inbounds secondary_1[i] += 1i32
        end

        # check when a certain number of attempts have been hit and main is below a certain level
        # check if goal is reachable
        # check if no attempts left and goal has not been reached
        # check if main is lower than level 8
        reset_check = 1i32 * ((attempt == reset_at_attempt) && (secondary_1[i] < reset_below_level) && (secondary_2[i] < reset_below_level))
        reset_check += 1i32 * (((20i32 - attempt) < (goal_level - secondary_1[i])) && ((20i32 - attempt) < (goal_level - secondary_2[i])))
        reset_check += 1i32 * ((attempt == 20i32) && (secondary_1[i] < goal_level) && (secondary_2[i] < goal_level))
        reset_check *= 1i32 * (main[i] < 8)

        if reset_check >= 1
            @inbounds reset[i] += 1i32
            attempt = 0i32

            @inbounds main[i] = 0i32
            @inbounds secondary_1[i] = 0i32
            @inbounds secondary_2[i] = 0i32
        end
    end
    
    return
end


probability_main = [0.35, 0.35, 0.35, 0.2, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05, -1]                  
cost_data = [10, 10, 10, 20, 20, 20, 30, 30, 40, 50, 0]

simulate_main = false
goal_level = 8

attempt_magnitude = 10 # Does a total of roughly 10^n_magnitude attempts 
n_threads = 1024 # Recommended to be a multiple of 32. Lower this value when you want lower n_magnitude

folder_results = "results"

if simulate_main
    sub_folder = "main"
else
    sub_folder = "secondary"
end

save_path = "./$(folder_results)/$(sub_folder)"

if !isdir(save_path)
    mkdir(save_path)
end

summary_cost_mean = zeros(Int32, 10, goal_level)
summary_cost_50 = zeros(Int32, 10, goal_level)
summary_cost_90 = zeros(Int32, 10, goal_level)
summary_cost_95 = zeros(Int32, 10, goal_level)
summary_cost_99 = zeros(Int32, 10, goal_level)

summary_reset_mean = zeros(Int32, 10, goal_level)
summary_reset_50 = zeros(Int32, 10, goal_level)
summary_reset_90 = zeros(Int32, 10, goal_level)
summary_reset_95 = zeros(Int32, 10, goal_level)
summary_reset_99 = zeros(Int32, 10, goal_level)

for (i, reset_at_attempt) in enumerate(10:19)
    reset_below_level_minimum = maximum((0, goal_level - (20 - reset_at_attempt)))
    for reset_below_level in reset_below_level_minimum:(goal_level-1)
        println("-"^50)
        if ~simulate_main
            println("Goal for main = 8")
        end
        println("Goal for $(sub_folder) = ", goal_level)
        println("Reset at attempt = ", reset_at_attempt)
        println("Reset when $(sub_folder) below level = ", reset_below_level)

        # Approximate number of resets to tune number of attempts
        main_cuda, secondary_1_cuda, secondary_2_cuda, 
        cost_cuda, reset_cuda = compute_hexa_reset(1024, probability_main, cost_data, 
                                                   goal_level, reset_at_attempt, reset_below_level,
                                                   n_threads, simulate_main);

        reset = Array(reset_cuda)
        reset_mean = mean(reset)
        magnitude = Int32(ceil(log2(10^(attempt_magnitude-round(log10(reset_mean))))))

        # Comment this for lower amount of trials
        if magnitude < 10
            magnitude = 10
        elseif magnitude > 27
            magnitude = 27
        end

        println(f"\nNumber of trials : {2^magnitude:d}")

        main_cuda, secondary_1_cuda, secondary_2_cuda, 
        cost_cuda, reset_cuda = compute_hexa_reset(2^magnitude, probability_main, cost_data, 
                                                   goal_level, reset_at_attempt, reset_below_level,
                                                   n_threads, simulate_main);

        main = Array(main_cuda)
        secondary_1 = Array(secondary_1_cuda)
        secondary_2 = Array(secondary_2_cuda)
        cost = Array(cost_cuda)
        reset = Array(reset_cuda)

        summary_cost_mean[i, reset_below_level+1] = ceil(mean(cost))
        summary_cost_50[i, reset_below_level+1] = ceil(quantile(cost, .50, sorted=false, alpha=0, beta=1))
        summary_cost_90[i, reset_below_level+1] = ceil(quantile(cost, .90, sorted=false, alpha=0, beta=1))
        summary_cost_95[i, reset_below_level+1] = ceil(quantile(cost, .95, sorted=false, alpha=0, beta=1))
        summary_cost_99[i, reset_below_level+1] = ceil(quantile(cost, .99, sorted=false, alpha=0, beta=1))

        summary_reset_mean[i, reset_below_level+1] = ceil(mean(reset))
        summary_reset_50[i, reset_below_level+1] = ceil(quantile(reset, .50, sorted=false, alpha=0, beta=1))
        summary_reset_90[i, reset_below_level+1] = ceil(quantile(reset, .90, sorted=false, alpha=0, beta=1))
        summary_reset_95[i, reset_below_level+1] = ceil(quantile(reset, .95, sorted=false, alpha=0, beta=1))
        summary_reset_99[i, reset_below_level+1] = ceil(quantile(reset, .99, sorted=false, alpha=0, beta=1))

        println(f"\ncost mean  : {summary_cost_mean[i, reset_below_level+1]:d}")
        println(f"cost q=50% : {summary_cost_50[i, reset_below_level+1]:d}")
        println(f"cost q=90% : {summary_cost_90[i, reset_below_level+1]:d}")
        println(f"cost q=95% : {summary_cost_95[i, reset_below_level+1]:d}")
        println(f"cost q=99% : {summary_cost_99[i, reset_below_level+1]:d}")

        println(f"\nreset mean  : {summary_reset_mean[i, reset_below_level+1]:d}")
        println(f"reset q=50% : {summary_reset_50[i, reset_below_level+1]:d}")
        println(f"reset q=90% : {summary_reset_90[i, reset_below_level+1]:d}")
        println(f"reset q=95% : {summary_reset_95[i, reset_below_level+1]:d}")
        println(f"reset q=99% : {summary_reset_99[i, reset_below_level+1]:d}")
    end
end

writedlm(save_path * "/summary_cost_mean_$(goal_level).txt", summary_cost_mean)
writedlm(save_path * "summary_cost_50_$(goal_level).txt", summary_cost_50)
writedlm(save_path * "/summary_cost_90_$(goal_level).txt", summary_cost_90)
writedlm(save_path * "/summary_cost_95_$(goal_level).txt", summary_cost_95)
writedlm(save_path * "/summary_cost_99_$(goal_level).txt", summary_cost_99)

writedlm(save_path * "/summary_reset_mean_$(goal_level).txt", summary_reset_mean)
writedlm(save_path * "/summary_reset_50_$(goal_level).txt", summary_reset_50)
writedlm(save_path * "/summary_reset_90_$(goal_level).txt", summary_reset_90)
writedlm(save_path * "/summary_reset_95_$(goal_level).txt", summary_reset_95)
writedlm(save_path * "/summary_reset_99_$(goal_level).txt", summary_reset_99)
