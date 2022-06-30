function rotated_vec = rotate_vec_using_quat(vec,rot_vec)

rotated_vec = compact(prod([prod([rot_vec;quaternion([0 vec])]);conj(rot_vec)]));
rotated_vec = rotated_vec(2:4);
end
