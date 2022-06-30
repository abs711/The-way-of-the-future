function quat = eul2quat_ZXY(q)
warning('ROWS REPRESENT A TIME INSTANT AND COLUMNS REPRESENT AXES')
warning('INPUT VECTOR IN THE SEQUENCE [qx qy qz]')
quat1 = -sin(q(:,3)/2).*sin(q(:,1)/2).*sin(q(:,2)/2)+cos(q(:,3)/2).*cos(q(:,1)/2).*cos(q(:,2)/2);
quat2 = -sin(q(:,3)/2).*sin(q(:,2)/2).*cos(q(:,1)/2)+sin(q(:,1)/2).*cos(q(:,3)/2).*cos(q(:,2)/2);
quat3 = +sin(q(:,3)/2).*sin(q(:,1)/2).*cos(q(:,2)/2)+sin(q(:,2)/2).*cos(q(:,3)/2).*cos(q(:,1)/2);
quat4 = +sin(q(:,3)/2).*cos(q(:,1)/2).*cos(q(:,2)/2)+sin(q(:,1)/2).*cos(q(:,2)/2).*cos(q(:,3)/2);
quat = quaternion([quat1, quat2, quat3, quat4]);
end