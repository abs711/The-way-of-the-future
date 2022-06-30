function new_rot_vec = get_rotations(rot_vec,rotation_true)

if rotation_true
    new_rot_vec = quaternion(rot_vec(1:4));
    for i = 2:23
        new_rot_vec = [new_rot_vec ; quaternion(rot_vec(4*i-3:4*i))];
    end
else
    new_rot_vec = quaternion([1 0 0 0])*ones(1,23);
end
end