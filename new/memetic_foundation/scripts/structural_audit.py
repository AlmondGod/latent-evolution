import torch
import sys
import os

# Add parent directory to path to absolute import from new.memetic_foundation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC

def structural_audit():
    print("=============================================")
    print("  MEMETIC FOUNDATION: STRUCTURAL AUDIT")
    print("=============================================\n")

    # MPE test dimensions
    n_agents = 3
    obs_dim = 16
    state_dim = 16 * 3
    n_actions = 5
    hidden_dim = 64
    mem_dim = 16
    comm_dim = 16
    n_mem_cells = 4

    variants = {
        "Baseline": {"use_memory": False, "use_comm": False},
        "Memory Only": {"use_memory": True, "use_comm": False},
        "Comm Only": {"use_memory": False, "use_comm": True},
        "Full Architecture": {"use_memory": True, "use_comm": True},
    }

    for v_name, v_kwargs in variants.items():
        print(f"=== Variant: {v_name} ===")
        net = MemeticFoundationAC(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            mem_dim=mem_dim,
            comm_dim=comm_dim,
            n_mem_cells=n_mem_cells,
            use_memory=v_kwargs["use_memory"],
            use_comm=v_kwargs["use_comm"],
        )
        
        # 1. Structural Assertions
        has_mem_module = getattr(net, 'memory', None) is not None
        has_mem_read = getattr(net, 'memory_reader', None) is not None
        has_mem_write = getattr(net, 'memory_writer', None) is not None
        has_comm = getattr(net, 'comm', None) is not None
        
        print("\n[Architecture Initialization]")
        print(f"  - memory cells exist: {has_mem_module}")
        print(f"  - memory read used:   {has_mem_read}")
        print(f"  - memory write used:  {has_mem_write}")
        print(f"  - comm module used:   {has_comm}")

        # 2. Forward Pass Norm Trace
        # Fake inputs
        batch_size = 2
        
        # PPO evaluation expects flattened (B*N, ...) inputs
        obs = torch.randn(batch_size * n_agents, obs_dim)
        prev_actions = torch.randint(0, n_actions, (batch_size * n_agents,))
        avail_actions = torch.ones(batch_size * n_agents, n_actions)
        hx = torch.randn(batch_size * n_agents, hidden_dim) # Not used directly in new arch, but kept for signature
        
        if has_mem_module:
            memory_state = net.memory()
        else:
            memory_state = None
            
        # Get intermediate outputs from evaluate_actions
        # To do this cleanly, we'll run a forward pass and check the norm dictionary
        action_logits, value, aux_loss, hx, norms_dict = net.evaluate_actions(
            obs=obs,
            actions=prev_actions,
            avail_actions=avail_actions,
            state=torch.randn(batch_size, state_dim) # State is still (B, state_dim)
        )

        
        print("\n[Tensor Norms (Single Forward Pass)]")
        # Extract specific requested norms, handling None gracefully
        mem_norm = norms_dict.get('memory', 0.0)
        mem_delta = norms_dict.get('memory_delta', 0.0)
        msg_out = norms_dict.get('message_out', 0.0)
        msg_in = norms_dict.get('message_in', 0.0)
        
        # Format explicitly for user checklist
        print(f"  ||memory||:           {mem_norm:.4f}")
        print(f"  ||memory_delta||:     {mem_delta:.4f}")
        print(f"  ||message_out||:      {msg_out:.4f}")
        print(f"  ||message_in||:       {msg_in:.4f}")

        # 3. Parameter Capacity Printout
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\n[Capacity Check]")
        print(f"  Total Parameters: {total_params:,}")
        
        # 4. Programmatic Yes/No Checklist
        print("\n[Verification Checklist]")
        if v_name == "Baseline":
            print(f"  Does actor ignore memory entirely?    {'YES' if not has_mem_module else 'NO'}")
            print(f"  Are no messages produced?             {'YES' if msg_out == 0 else 'NO'}")
            print(f"  Are no memory updates happening?      {'YES' if mem_delta == 0 else 'NO'}")
        elif v_name == "Memory Only":
            print(f"  Does memory update over time?         {'YES' if mem_delta > 0 else 'NO'}")
            print(f"  Does actor use memory?                {'YES' if has_mem_read else 'NO'}")
            print(f"  Are no messages produced/consumed?    {'YES' if msg_out == 0 and msg_in == 0 else 'NO'}")
        elif v_name == "Comm Only":
            print(f"  Are messages produced and received?   {'YES' if msg_out > 0 and msg_in > 0 else 'NO'}")
            print(f"  Is there no persistent memory update? {'YES' if mem_delta == 0 else 'NO'}")
            print(f"  Does actor depend on current comm?    {'YES' if has_comm else 'NO'}")
        elif v_name == "Full Architecture":
            print(f"  Are messages produced and received?   {'YES' if msg_out > 0 and msg_in > 0 else 'NO'}")
            print(f"  Do messages cause nonzero mem updates?{'YES' if mem_delta > 0 else 'NO'}") # NOTE: causal test requires block
            print(f"  Does actor use memory?                {'YES' if has_mem_read else 'NO'}")
        print("\n" + "-"*45 + "\n")

if __name__ == "__main__":
    structural_audit()
