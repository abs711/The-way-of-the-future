function acts = compute_quaternion_actions(q)
acts = conj(q(1:end-1,:)).*q(2:end,:);
end