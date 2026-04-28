# Zigzag pattern

_W3 = np.pi / 2   # scoop orientation

# Hover above sand
_Q_ABOVE = np.array([-0.27, -1.50,  1.70,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)

# Zig-zag corners: shoulder_pan varies L↔R; shoulder_lift+elbow are coupled
# to change reach depth (near vs far from base) while keeping EE height roughly
# constant (~±3 mm across near/far).
#   near: lift=-0.45, elbow=0.90 → arc closer to robot base
#   far:  lift=-0.55, elbow=1.10 → arc farther from robot base
_Q_LEFT_NEAR  = np.array([-0.55, -0.45,  0.90,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)
_Q_RIGHT_FAR  = np.array([ 0.20, -0.55,  1.10,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)
_Q_LEFT_FAR   = np.array([-0.55, -0.55,  1.10,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)
_Q_RIGHT_NEAR = np.array([ 0.20, -0.45,  0.90,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)

# Waypoint list: (target_joint_angles, duration_seconds)
# The scoop traces a zig-zag across the pile:
#   diagonal zig  : LEFT_NEAR → RIGHT_FAR  (pan + reach change together)
#   lateral reset : RIGHT_FAR → LEFT_FAR   (pan only, staying deep)
#   diagonal zag  : LEFT_FAR  → RIGHT_NEAR (pan + reach change together)
#   lateral reset : RIGHT_NEAR → LEFT_NEAR (pan only, staying near)
_WAYPOINTS = [
    (_Q_ABOVE,      2.5),  # hover — let sand settle under gravity
    (_Q_LEFT_NEAR,  1.5),  # descend into sand at near-left corner
    (_Q_RIGHT_FAR,  2.5),  # diagonal zig → far-right
    (_Q_LEFT_FAR,   2.0),  # lateral reset ← far-left
    (_Q_RIGHT_NEAR, 2.5),  # diagonal zag → near-right
    (_Q_LEFT_NEAR,  2.0),  # lateral reset ← near-left
    (_Q_RIGHT_FAR,  2.5),  # repeat zig
    (_Q_LEFT_FAR,   2.0),
    (_Q_RIGHT_NEAR, 2.5),  # repeat zag
    (_Q_LEFT_NEAR,  2.0),
    (_Q_ABOVE,      1.5),  # lift out — loop repeats
]

#---