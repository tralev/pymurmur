# Boids Simulation Math

## Contents

| § | Platform | Key Techniques |
|:-:|----------|---------------|
| 01 | Vanilla JS | Core Reynolds (cohesion, separation, alignment) |
| 02 | JS Canvas | `steer = desired − velocity`, 1/r² separation, obstacle avoidance |
| 03 | Panda3D (Python) | Seek & Arrive steering, ray-plane mouse projection |
| 04 | C + Raylib | Rotation-based movement, priority-zone rule selection |
| 05 | Unity GPU (HLSL) | 3D Simplex noise, obstacle hard-override, zone containment |
| 06 | Processing (Java) | Fixed-speed, spherical confinement, 7-neighbor cap |
| 07 | Processing (Java) | Metric-radius sliders & k-NN topology with roosting |
| 08 | WebGL/Three.js | Ping-pong textures, cosine-weighted three-zone curves, predator |
| 09 | Scala 2D | Velocity-inertia blending, see-ahead ray obstacle avoidance |
| 10 | Unity GPU (HLSL) | O(n²) all-pairs & parallel prefix-sum reduction variants |
| 11 | Unity 3D | FOV cone, state machine, skeletal trails, predator waves |
| 12 | Unity GPU (HLSL) | CPU/GPU Reynolds, 1/d wall force, reflective boundaries |
| 13 | Blazor (C#) | Exponential separation, 1/d weighted cohesion, linear walls |
| 14 | Unity ECS (multi) | FOV neighbor detection, direct position-difference cohesion |
| 15 | Unity DOTS + Burst | Priority-ordered force stack, wander, banking, damping |
| 16 | Rust + Nannou | Angle-based movement, circular mean, edge state machine |
| 17 | Python + Pygame | Angle-based, 7-nearest, spatial grid, pixel trail variants |
| 18 | Rust + Bevy | Reynolds steering, golden-spiral raycast, Lissajous target |
| 19 | Godot (skeleton) | MultiMesh renderer only — no flocking math implemented |
| 20 | Unreal Engine 5 | Hashed grid + bitonic sort, exponential heading smoothing |

## 01

This simulation implements Craig Reynolds' Boids flocking algorithm with three core steering behaviors.

### Constants

- `numBoids = 100` — number of boids
- `visualRange = 75` — perception radius for neighbors
- `minDistance = 20` — separation distance threshold
- `centeringFactor = 0.005` — cohesion weight
- `avoidFactor = 0.05` — separation weight
- `matchingFactor = 0.05` — alignment weight
- `turnFactor = 1` — boundary turning force
- `speedLimit = 15` — maximum speed
- `margin = 200` — distance from edge to begin turning

### Euclidean Distance

The distance between two boids is computed as:

```
d(boid₁, boid₂) = √[(boid₁.x − boid₂.x)² + (boid₁.y − boid₂.y)²]
```

### Rule 1: Cohesion (flyTowardsCenter)

Steer toward the average position (center of mass) of all neighbors within `visualRange`.

Given a boid *B* and its *N* neighbors within the visual range:

```
centerX = (1/N) × Σ neighbor.x
centerY = (1/N) × Σ neighbor.y

B.dx += (centerX − B.x) × centeringFactor
B.dy += (centerY − B.y) × centeringFactor
```

### Rule 2: Separation (avoidOthers)

Steer away from neighbors that are too close (within `minDistance`).

```
moveX = Σ (B.x − neighbor.x)    for all neighbors where d(B, neighbor) < minDistance
moveY = Σ (B.y − neighbor.y)    for all neighbors where d(B, neighbor) < minDistance

B.dx += moveX × avoidFactor
B.dy += moveY × avoidFactor
```

### Rule 3: Alignment (matchVelocity)

Match the average velocity (direction and speed) of neighbors within `visualRange`.

Given *N* neighbors within the visual range:

```
avgDX = (1/N) × Σ neighbor.dx
avgDY = (1/N) × Σ neighbor.dy

B.dx += (avgDX − B.dx) × matchingFactor
B.dy += (avgDY − B.dy) × matchingFactor
```

### Speed Limiting (limitSpeed)

```
speed = √(B.dx² + B.dy²)

if speed > speedLimit:
    B.dx = (B.dx / speed) × speedLimit
    B.dy = (B.dy / speed) × speedLimit
```

### Boundary Handling (keepWithinBounds)

Steer back toward the center when too close to an edge:

```
if B.x < margin:          B.dx += turnFactor
if B.x > width − margin:  B.dx -= turnFactor
if B.y < margin:          B.dy += turnFactor
if B.y > height − margin: B.dy -= turnFactor
```

### Position Update

```
B.x += B.dx
B.y += B.dy
```

### Drawing Angle

```
angle = atan2(B.dy, B.dx)
```

Each boid is rotated by this angle so its triangle shape points in the direction of motion.

### Initialization

Boids are initialized with random positions and velocities:

```
B.x  = rand(0, width)
B.y  = rand(0, height)
B.dx = rand(−5, 5)
B.dy = rand(−5, 5)
```

## 02

This simulation uses proper Craig Reynolds steering forces with a **steer = desired − velocity** formulation, capped by a maximum force for natural turning. It also adds obstacle avoidance via orthogonal projection and 1/r² separation falloff.

### Constants

- `numBoids = 70` — number of boids
- `maxSpeed = 6` — maximum speed of each boid
- `maxForce = 0.03` — maximum steering force (kept low for natural behavior)
- `defaultVelocity = 3` — initial speed magnitude
- `followRadius = 100` — perception radius for cohesion and alignment
- `avoidRadius = 30` — separation distance threshold
- `obstacleAffectRadius = 100` — obstacle influence radius
- Weight coefficients: `cohesion = 1`, `avoidance = 4`, `following = 2`, `obstacles = 0.4`

### Euclidean Distance

```
displacement = neighbor.position − B.position
|displacement| = √(displacement.x² + displacement.y²)
```

### Steering Force (getSteer)

Given a **desired** velocity vector, compute the steering force needed to turn toward it, capped at `maxForce`:

```
if |desired| = 0: steer = (0, 0)
else:            |desired| = maxSpeed

steer = desired − B.velocity

if |steer| > maxForce:
    |steer| = maxForce
```

### Rule 1: Cohesion

Steer toward the center of mass of neighbors within `followRadius`.

```
centerOfMass = (1/N) × Σ neighbor.position    for all neighbors where |displacement| < followRadius

desired = centerOfMass − B.position
steer   = getSteer(desired)
```

### Rule 2: Separation

Steer away from neighbors within `avoidRadius` using a 1/r² falloff. Each neighbor contributes a repulsion vector of `−r̂ / |r|²`:

```
for each neighbor ≠ B where |displacement| < avoidRadius:
    avoidanceVector −= normalize(displacement) / |displacement|

steer = getSteer(avoidanceVector)
```

The `normalize(displacement) / |displacement|` term equals `displacement / |displacement|²`, i.e. the repulsion decays quadratically with distance.

### Rule 3: Alignment

Match the average velocity of neighbors within `followRadius`.

```
avgVelocity = (1/N) × Σ neighbor.velocity    for all neighbors ≠ B where |displacement| < followRadius

steer = getSteer(avgVelocity)
```

### Obstacle Avoidance

Boids avoid obstacles using orthogonal projection onto their velocity vector. For each obstacle:

```
disp = obstacle.position − B.position
perp = disp − B.velocity × (disp · B.velocity) / |B.velocity|²

if |perp| < obstacle.affectRadius:
    keep the closest obstacle's −perp as the obstacle vector

if |obstacleVector| > 0:
    obstacleVector = normalize(obstacleVector)
```

The `perp` vector is the component of the displacement orthogonal to the boid's forward direction — the boid only steers away if the perpendicular miss distance is within the obstacle's affect radius.

### Velocity Update

All forces are weighted, summed, and applied. Random noise is also added:

```
noise = (rand(−0.5, 0.5), rand(−0.5, 0.5)) × 2 × 0.1

B.velocity += cohesion × weight_cohesion
B.velocity += avoidance × weight_avoidance
B.velocity += follow    × weight_following
B.velocity += obstacle  × weight_obstacles
B.velocity += noise
```

### Speed Limiting

```
if |B.velocity| > maxSpeed:
    |B.velocity| = maxSpeed
```

This is a direct clamp (truncation) rather than normalization — the direction is preserved and the magnitude is simply capped.

### Position Update

```
B.rotation = atan2(B.velocity.y, B.velocity.x)
B.position += B.velocity
```

### Boundary Handling

Wrap-around (toroidal) instead of steering back:

```
if B.x > width:  B.x = 0
if B.x < 0:      B.x = width
if B.y > height: B.y = 0
if B.y < 0:      B.y = height
```

### Initialization

```
B.position = randomPoint × viewSize           // uniform random within canvas
B.velocity = (|v| = defaultVelocity, angle = rand(0°, 360°))
```

### Dynamic Color

Each boid's color hue is determined by its direction of motion:

```
hue = atan2(B.velocity.y, B.velocity.x)
```

## 03

This is a Panda3D 3D demonstration of two autonomous steering behaviors — **Seek** and **Arrive** — where two boids chase a user-controlled target. It is not a full flocking simulation; instead it isolates the core steering math.

### Constants

| Boid  | Behavior | maxForce | maxSpeed |
|-------|----------|----------|----------|
| Red   | Seek     | 4.0      | 0.1      |
| Blue  | Arrive   | 4.0      | 1.0      |

- `arriveRadius = 100.0` — distance at which the Arrive boid begins to slow down
- `radius = 2.0` — boid radius (3D)

### Steering Force

This is the core of both Seek and Arrive. Given a target position and a slow-down flag:

```
desired = target − B.location
|desired| = √(desired.x² + desired.y² + desired.z²)

if |desired| > 1.0:
    d̂ = desired / |desired|                         // normalize

    if slowDown and |desired| < arriveRadius:       // ARRIVE: scale speed by proximity
        desired = d̂ × maxSpeed × (|desired| / arriveRadius)
    else:                                           // SEEK: full speed
        desired = d̂ × maxSpeed

    steer = desired − B.velocity

    if |steer| > maxForce:
        steer = (steer / |steer|) × maxForce        // clamp to max force
else:
    steer = (0, 0, 0)
```

The key difference between Seek and Arrive:
- **Seek** always targets `d̂ × maxSpeed`, so it barrels toward the target at full speed.
- **Arrive** scales the desired speed linearly with distance when within `arriveRadius`: `desired speed = maxSpeed × (|desired| / arriveRadius)`. This causes the boid to decelerate smoothly as it approaches the target, reaching zero speed at distance zero.

### Seek Behavior

```
B.acceleration += steer(target, slowDown = false)
B.lookAt(target)
```

### Arrive Behavior

```
B.acceleration += steer(target, slowDown = true)
B.lookAt(target)
```

### Velocity Update

```
B.velocity += B.acceleration

if |B.velocity| > maxSpeed:
    B.velocity = (B.velocity / |B.velocity|) × maxSpeed    // normalize then scale
```

### Position Update

```
B.location += B.velocity
B.acceleration = (0, 0, 0)    // reset each frame
```

### Mouse-to-World Projection

The target is moved by casting a ray from the camera through the mouse cursor and intersecting it with a plane parallel to the camera at z = 0:

```
Ray: r⃗(t) = cameraPos + t × directionFromLens(mouseX, mouseY)
Plane: z = 0, parallel to camera view plane

intersection = ray ∩ plane → surfacePoint
```

### Initialization

```
B.location = (0, 0, 0)
B.velocity = (0, 0, 0)
B.acceleration = (0, 0, 0)
```

## 04

A C + Raylib implementation using **rotation-based movement**. Each boid has a scalar rotation angle and a speed magnitude. Movement is computed by projecting the rotation into Cartesian components. Flocking rules use a distance-based priority system rather than weighted force summation.

### Constants

- `numBoids = 128` — number of boids
- `neighborRadius = 50` — perception radius (local flock)
- `maxLocalFlock = 128` — maximum neighbors tracked
- `velocityMagnitude = (20, 20)` — speed in pixels/second
- `angularVelocity = 1` — maximum turn rate in radians/second
- `screenWidth = 800`, `screenHeight = 450`
- `separationThreshold = 5` — distance below which separation triggers
- `cohesionThreshold = 30` — distance above which cohesion kicks in
- `alignmentRange = (10, 30)` — alignment is the default in between

### Euclidean Distance

```
d(v₁, v₂) = √(|v₁.x − v₂.x|² + |v₁.y − v₂.y|²)
```

### Angle Between Two Points (getRotation)

Returns the angle from point v₂ toward point v₁:

```
δ = (v₁.x − v₂.x, v₁.y − v₂.y)
angle = atan2(−δ.x, δ.y)
```

### Rotation to Velocity

Velocity is derived from the boid's rotation angle and speed magnitude:

```
vx = sin(θ) × speed.x
vy = −cos(θ) × speed.y
```

### 2D Rotation of Boid Vertices (rotateBoid)

Each boid's triangle vertices are rotated by angle θ using the standard rotation matrix:

```
[x′]   [cos θ   −sin θ] [x]
[y′] = [sin θ    cos θ] [y]

x′ = cos(θ) × x − sin(θ) × y
y′ = sin(θ) × x + cos(θ) × y

B.rotation = fmod(B.rotation + θ, 2π)
```

### Angle Utilities

```
MODULO(a, n) = fmod(a, n) + ((a < 0) × n)    // true mathematical modulo; always non-negative
INVERSE(θ)   = fmod(θ + π, 2π)                // opposite direction
```

### Cohesion (getCohesion)

Compute the center of mass of neighbors within `neighborRadius`, then return the angle pointing toward it:

```
if N = 0: return B.rotation

mean = (1/N) × Σ neighbor.origin
return getRotation(B.origin, mean)
```

### Alignment (getAlignment)

Average the rotation angles of all neighbors:

```
if N = 0: return B.rotation

return (1/N) × Σ neighbor.rotation
```

### Separation (getSeparation)

Find the closest neighbor. If within `separationThreshold`, steer in the opposite direction:

```
closestDistance = min( d(B, neighbor) )  for all neighbors
closestAngle    = getRotation(B.origin, closestNeighbor.origin)

if closestDistance > 5: return B.rotation       // no separation needed
else:                   return INVERSE(closestAngle)   // turn away
```

### Priority-Based Rule Selection (updateBoid)

Instead of blending all three rules, a single rule is chosen based on the distance to the nearest neighbor. This creates three behavioral zones:

```
targetRotation = alignment − B.rotation    // default: align

if closestDistance ≥ 30:
    targetRotation = cohesion − B.rotation   // far → cohere

if closestDistance ≤ 10:
    targetRotation = separation − B.rotation // close → separate
```

| Zone            | Distance d    | Behavior    |
|-----------------|---------------|-------------|
| Separation zone | d ≤ 10        | Turn away   |
| Alignment zone  | 10 < d < 30   | Match angle |
| Cohesion zone   | d ≥ 30        | Steer toward center |

### Rotation Clamping

Take the shortest angular path and clamp turn rate by `angularVelocity × Δt`:

```
targetRotation = MODULO(targetRotation, 2π)

if targetRotation > π:
    targetRotation = INVERSE(targetRotation) − π    // shorter path

maxTurn = angularVelocity × Δt

targetRotation = clamp(targetRotation, −maxTurn, maxTurn)

rotateBoid(B, targetRotation)
```

### Position Update

```
vx = sin(B.rotation) × B.speed.x
vy = −cos(B.rotation) × B.speed.y

B.origin.x = MODULO(B.origin.x + vx × Δt, 800)    // wrap-around
B.origin.y = MODULO(B.origin.y + vy × Δt, 450)
```

### Initialization

```
B.origin    = (rand(0, 800), rand(0, 450))
B.speed     = (20, 20)
B.rotation  = rand(0, 6)    // radians
B.angularV  = 1

Triangle vertices relative to origin:
    v₀ = (0, −5)     // nose
    v₁ = (−5, 5)     // left wing
    v₂ = (5, 5)      // right wing
```

## 05

A Unity GPU compute shader implementation running thousands of boids in parallel. It uses **3D Simplex noise** for turbulent speed fields, **obstacle avoidance with hard heading override**, **attraction/repulsion targets**, and **spatial zone containment** (slab, box, sphere). No explicit cohesion — instead target attraction replaces it. Flocking is O(n²) per boid.

### Constants

- `boidCount = 1000` — default number of boids (scalable)
- `moveSpeed = 5` — base movement speed
- `cellRadius = 5` — perception radius for neighbors
- `separationWeight = 5` — separation force multiplier
- `alignmentWeight = 2` — alignment force multiplier
- `targetWeight = 3` — global target-force multiplier
- `obstacleAversionDistance = 3` — distance threshold from obstacle surface
- `noiseScale = 0.1` — spatial frequency of noise field
- `noiseScrollSpeed = (0.5, 0, 0.5)` — direction and speed of noise advection
- `minZoneSpeed = 0.5`, `maxZoneSpeed = 2.0` — noise-driven speed range
- `turbulencePower = 3.0` — sharpens transitions between slow/fast zones

### Euclidean Distance

```
d(P, Q) = √((P.x − Q.x)² + (P.y − Q.y)² + (P.z − Q.z)²)
```

### Separation

For all neighbors within `cellRadius`, sum their positions. The separation vector points away from their center of mass:

```
separationSum  = Σ neighbor.position      for all neighbors where d(B, neighbor) < cellRadius
alignmentSum   = Σ neighbor.direction      for all same neighbors
N = neighborCount

separationVector = (B.position × N) − separationSum
separationResult = separationWeight × normalize(separationVector)
```

This is equivalent to: `separationResult = separationWeight × normalize(B.position − avgPosition)`, i.e. steer away from the average neighbor position.

### Alignment

```
avgAlignment   = alignmentSum / N
alignmentResult = alignmentWeight × normalize(avgAlignment − B.forward)
```

Steers toward the average forward direction of neighbors.

### Target Attraction/Repulsion

Each force provider has a 3D position and a weight (positive = attraction, negative = repulsion):

```
for each target k:
    dirToTarget = target.position − B.position
    dist = |dirToTarget|
    if dist > 0.001:
        d̂ = dirToTarget / dist
        targetForce += d̂ × target.weight

targetForce *= targetWeight
```

### Heading Blending

Separation, alignment, and target forces are summed and normalized into a single desired heading:

```
normalHeading = normalize(alignmentResult + separationResult + targetForce)
```

### Obstacle Avoidance (Hard Override)

Find the nearest obstacle by distance, then if the boid is within its influence zone, override the heading to steer directly away from the obstacle surface:

```
nearest = argmin( d(B.position, obstacle.position) ) over all obstacles

distToSurface = d(B.position, nearest.position) − nearest.radius

if distToSurface < obstacleAversionDistance:
    avoidDir    = normalize(B.position − nearest.position)
    avoidPoint  = nearest.position + avoidDir × (nearest.radius + obstacleAversionDistance)
    finalHeading = normalize(avoidPoint − B.position)
else:
    finalHeading = normalHeading
```

### 3D Simplex Noise (snoise)

Used to create a turbulent speed field. The boid's position is transformed into noise space and time-scrolled:

```
noisePos  = (B.position × noiseScale) + (time × noiseScrollSpeed)
noiseVal  = snoise(noisePos)                          // ∈ [−1, 1]
t         = ((noiseVal + 1) × 0.5)^turbulencePower   // ∈ [0, 1], sharpened

currentZoneSpeed = lerp(minZoneSpeed, maxZoneSpeed, t)
```

Raising to `turbulencePower` creates sharper transitions between slow pockets and fast surges.

### Heading & Velocity Integration

The heading smoothly interpolates toward the desired heading, then velocity is computed with the noise-modulated speed:

```
if |finalHeading| > 0.001:
    nextHeading = normalize(B.forward + Δt × (finalHeading − B.forward))
    B.forward   = nextHeading

    velocity = nextHeading × (moveSpeed × currentZoneSpeed)
    B.position += velocity × Δt
```

This is exponential smoothing toward `finalHeading` with rate `Δt` (no separate turn-rate parameter).

### Spatial Zone Containment

After movement, boid positions are clamped inside containment zones defined in local space:

```
localPos = worldToLocal × (B.position, 1)

// Slab (type 0): clamp Y only
halfThickness = dimensions.y / 2
localPos.y = clamp(localPos.y, −halfThickness, halfThickness)

// Box (type 1): clamp all axes
halfDim = dimensions / 2
if any(|localPos| > halfDim):
    localPos = clamp(localPos, −halfDim, halfDim)

// Sphere (type 2): clamp to radius
if |localPos| > dimensions.x:
    localPos = normalize(localPos) × dimensions.x

B.position = localToWorld × (localPos, 1)
```

### Initialization

```
// Sphere spawn:
B.position  = worldCenter + Random.insideUnitSphere × spawnRadius
B.direction = Random.onUnitSphere       // uniformly random on unit sphere

// Box spawn:
B.position  = worldCenter + rand(−boxSize/2, boxSize/2) per axis

// Circle spawn (flat XZ):
B.position  = worldCenter + (Random.insideUnitCircle × spawnRadius, 0)
```

## 06

A Processing (Java) 3D boid simulation with real-time sliders and physical measurements. It uses **fixed-speed movement** (velocity is always normalized to `FLIGHT_SPEED`), a **spherical confinement boundary** with priority avoidance, and a capped number of influencing neighbors (`INFLUENCE = 7`).

### Constants

- `INIT_FLOCK_SIZE = 300` — starting number of boids
- `FLIGHT_SPEED` — fixed speed (default 1.0, adjustable via up/down keys)
- `MAX_FORCE = 0.2` — maximum acceleration magnitude
- `INFLUENCE = 7` — max neighbor count for alignment/cohesion
- `INFLUENCE_CIRCLE = 80.0` — perception radius
- `MIN_SEP = 50.0` — separation distance threshold
- `RADIUS_OF_CONFINEMENT = 300.0` — spherical boundary radius
- `SCALE = 1.2` — rendering scale factor
- Slider defaults: `SEPARATION_FACTOR ≈ 3.0`, `COHESION_FACTOR ≈ 0.2`, `ALIGNMENT_FACTOR ≈ 0.02`, `AVOIDANCE_FACTOR = 0.05`, `NOISE_FACTOR = 0.05` (all have small random jitter)
- Physical scaling: `normalised_vel = 8.94 / FLIGHT_SPEED` (real starling speed ≈ 20 mph ≈ 8.94 m/s), `mass = 0.075` kg, `acc_peak = 40` m/s²

### Euclidean Distance

```
d(B, other) = |B.position − other.position|
```

### Separation

Repulsion with 1/d² falloff from all neighbors within `MIN_SEP`:

```
for each neighbor where 0 < d(B, neighbor) < MIN_SEP:
    r⃗ = B.position − neighbor.position
    B.acceleration += r⃗ × (SEPARATION_FACTOR / d²)
```

### Cohesion

Steer toward the average position of up to `INFLUENCE` neighbors within `INFLUENCE_CIRCLE`:

```
avgPosition = (1/N) × Σ neighbor.position     for up to INFLUENCE neighbors

desired = avgPosition − B.position
if |desired| > 0:
    B.acceleration += normalize(desired) × COHESION_FACTOR
```

### Alignment

Steer toward the average velocity of up to `INFLUENCE` neighbors within `0.75 × INFLUENCE_CIRCLE`:

```
avgVelocity = (1/N) × Σ neighbor.velocity     for up to INFLUENCE neighbors
                                           (only those within 0.75 × INFLUENCE_CIRCLE)

desired = avgVelocity − B.velocity
if |desired| > 0:
    B.acceleration += normalize(desired) × ALIGNMENT_FACTOR
```

Note: alignment uses a tighter radius (75% of `INFLUENCE_CIRCLE`) than cohesion. The same neighbor loop serves both rules — position averaging uses the full circle, velocity averaging uses the inner circle.

### Noise

```
randomDir = random3D()        // uniformly random unit vector
B.acceleration += randomDir × NOISE_FACTOR
```

### Acceleration Clamping

```
if |B.acceleration| > MAX_FORCE:
    B.acceleration = normalize(B.acceleration) × MAX_FORCE
```

### Wall Avoidance (Spherical Confinement)

A hard spherical boundary at `RADIUS_OF_CONFINEMENT`. When outside, acceleration is **not applied** and the boid is pushed inward:

```
if |B.position| > RADIUS_OF_CONFINEMENT:
    inward = normalize(B.position) × AVOIDANCE_FACTOR
    B.velocity −= inward
else:
    B.velocity += B.acceleration
```

This prioritizes boundary avoidance over all other behaviors.

### Velocity Normalization (Fixed Speed)

Boids always travel at exactly `FLIGHT_SPEED`:

```
if |B.velocity| > 0:
    B.velocity = normalize(B.velocity)

B.position += B.velocity × FLIGHT_SPEED
```

### Orientation (Rendering)

Boids are rotated to face their direction of motion using spherical-to-Euler conversion:

```
rotateY = atan2(−velocity.z, velocity.x)          // yaw
rotateZ = asin(velocity.y / |velocity|)            // pitch
```

### Initialization

```
B.acceleration = (0, 0, 0)
B.velocity     = random3D()                        // uniformly random unit vector
B.position     = random3D() × rand(0, RADIUS_OF_CONFINEMENT)
```

### Physical Measurements

```
Center of mass:       CoM = (1/N) × Σ B.position

Average speed:        (8.94 / FLIGHT_SPEED) × avg(|B.velocity|)
Average acceleration: 40 × avg(|B.acceleration|) / MAX_FORCE
Average dispersion:   avg(|B.position − CoM|)
Total power:          Σ |B.acceleration · B.velocity|   (scaled by normalisation constants)
Average power:        totalPower / N
Average ang. momentum: avg(|B.position × B.velocity|)
```

## 07

Two Processing (Java) 3D boid simulations from the same project. **fish_sliders** uses metric-radius neighborhoods with real-time sliders. **starlings_topology** uses **k-nearest-neighbor topology** (6 closest boids) and adds **roosting** behavior with per-boid asynchronous wing flapping.

### 7a — fish_sliders

Uses interactive sliders for real-time blending of separation, alignment, and cohesion. Wall avoidance uses a 1/d² weighted force on all 6 walls of a bounding box.

#### Constants

- `initBoidNum = 1000` — starting boid count
- `neighborhoodRadius = 100` — perception radius (metric)
- `maxSpeed = 2` — maximum velocity magnitude
- `maxSteerForce = 0.05` — maximum force on steering vectors
- `sc = 3` — render scale factor
- Slider weights: `sepWeight ∈ [0,10]`, `aliWeight ∈ [0,10]`, `cohWeight ∈ [0,10]`
- Bounding box: X ∈ [0, width], Y ∈ [0, height], Z ∈ [300, 900]

#### Separation

Repulsion with 1/d falloff (unit direction divided by distance):

```
for each neighbor where 0 < d ≤ neighborhoodRadius:
    r̂ = (B.pos − neighbor.pos) / d
    posSum += r̂

separation = posSum     // Σ r̂/d = Σ (B.pos − neighbor.pos) / d²
```

#### Alignment

Returns the raw average velocity of neighbors (clamped), not a steering vector:

```
avgVel = (1/N) × Σ neighbor.vel    for neighbors within neighborhoodRadius
alignment = limit(avgVel, maxSteerForce)
```

#### Cohesion

Steering vector toward the center of mass (XY distance only):

```
centerOfMass = (1/N) × Σ neighbor.pos    for neighbors where XY distance ≤ neighborhoodRadius
steer = centerOfMass − B.pos
cohesion = limit(steer, maxSteerForce)
```

#### Wall Avoidance

All 6 bounding-box walls repel with 1/d² weighting:

```
for each wall point W (top/bottom/left/right/front/back):
    steer = B.pos − W
    steer = steer / d(B.pos, W)²          // 1/r² falloff
    B.acc += steer × 5                    // wall weight
```

#### Flocking Combination

```
B.acc += alignment  × aliWeight
B.acc += cohesion   × cohWeight
B.acc += separation × sepWeight
```

#### Movement

```
B.vel += B.acc
B.vel = limit(B.vel, maxSpeed)
B.pos += B.vel
B.acc = 0
```

#### Boundary Wrap

```
if pos.x > width:   pos.x = 0
if pos.x < 0:       pos.x = width
if pos.y > height:  pos.y = 0
if pos.y < 0:       pos.y = height
if pos.z > 900:     pos.z = 300
if pos.z < 300:     pos.z = 900
```

#### Orientation

```
rotateY = atan2(−vel.z, vel.x)     // yaw
rotateZ = asin(vel.y / |vel|)      // pitch
```

#### Reynolds Steering (available but not used in flocking)

```
// Seek:
steer = target − pos
steer = limit(steer, maxSteerForce)

// Arrive:
offset = target − pos
d = |offset|
rampedSpeed = maxSpeed × (d / 100)
clippedSpeed = min(rampedSpeed, maxSpeed)
desiredVel = offset × (clippedSpeed / d)
steer = desiredVel − vel
```

#### Initialization

```
B.pos = (rand(0, 1500), rand(0, 1000), rand(0, 1000))
B.vel = (rand(−1, 1), rand(−1, 1), rand(1, −1))
B.acc = (0, 0, 0)
```

### 7b — starlings_topology

Uses **k-nearest-neighbor topology** (always exactly 6 neighbors) instead of metric radius. Adds **roosting** — a center-attraction force — and asynchronous wing flapping. Wall avoidance only repels from the ground and side/bounding walls.

#### Constants

- `initBoidNum = 400`
- `neighborhoodRadius = 400` — (used only if topology fails; 6-nearest always overrides)
- `maxSpeed = 2`, `maxSteerForce = 0.05`
- `topology_k = 6` — always exactly 6 closest boids
- `phase = rand(0.095, 0.14)` — per-boid wing flap frequency
- Flocking weights: `ali = 1`, `coh = 3`, `sep = 3`, `roost = 1`

#### Topological Neighbor Selection

Sort all boids by distance, take the 6 closest (excluding self):

```
neighbours = argsort(d(B, boid_i))[1:7]    // first 6 nearest (skip self)
```

#### Roosting Force

A gentle pull toward the center of the world (roost site). Decoupled into vertical and horizontal components:

```
roostPos = (width/2, height/2, 600)

// Vertical — gentle pull toward roost height:
steer.y = (roostPos.y − pos.y) × 0.0003

// Horizontal — constant-magnitude pull toward center XZ:
modPos     = (pos.x, 0, pos.z)
modPosDiff = roostPos − modPos     // drop Y
modPosDiff = normalize(modPosDiff) × 0.01
steer     += modPosDiff
```

#### Wall Avoidance

Only the ground is strongly avoided; other walls are weak:

```
accept += avoid(pos.x, 0, pos.z)     × 5     // ground: strong (1/d² × 5)
accept += avoid(width, pos.y, pos.z) × 0.1   // right
accept += avoid(0, pos.y, pos.z)     × 0.1   // left
accept += avoid(pos.x, pos.y, 300)   × 0.1   // front
accept += avoid(pos.x, pos.y, 900)   × 0.1   // back
```

The top (Y = height) is not avoided — boids can fly above the view.

#### Flocking Combination

```
B.acc += alignment  × 1
B.acc += cohesion   × 3
B.acc += separation × 3
B.acc += roosting   × 1
```

Cohesion and separation are weighted much higher than alignment and roosting.

#### Boundary Wrap (Y only)

```
if pos.y < 0: pos.y = height
```

X and Z are not wrapped — wall avoidance handles them.

#### Wing Flapping (Rendering)

```
t += phase                         // per-boid phase for asynchrony
flap = 10 × sin(t)                 // wing Z displacement
```

Each boid has its own `phase ∈ [0.095, 0.14]`, so wings beat at slightly different frequencies.

#### Orientation

```
rotateY = atan2(−vel.z, vel.x)
rotateZ = asin(vel.y / |vel|)
rotateX = −π/2                     // additional rotation (model-dependent)
```

#### Initialization

```
B.pos = (rand(width/4, 3×width/4), rand(0, height), rand(width/4, 3×width/4))
B.vel = (rand(−1, 1), rand(−1, 1), rand(1, −1))
B.acc = (0, 0, 0)
```

## 08

A WebGL/Three.js GPU simulation using **fragment shaders on ping-pong textures** — each pixel encodes one boid. The flocking rules use O(n²) pairwise comparisons with **three concentric spherical zones** using cosine-shaped weighting curves rather than linear blending. Includes a **predator (falcon)** that boids flee from.

### Constants

- `speed_limit = 10.0` — normal max speed
- `speed_limit_flee = 15.0` — max speed when fleeing predator
- `radii_falcon_view = 200.0` — predator detection radius
- `upper_bound = 800`, `lower_bound = −800` — world bounds
- `cohesion_distance`, `alignment_distance`, `seperation_distance` — tunable zone radii (set as uniforms)
- `radii_sector = (cohesion_distance + alignment_distance + seperation_distance) × 1.15` — total perception radius
- `area_sector = radii_sector²`
- `separation_limit = seperation_distance / radii_sector` — normalized threshold
- `alignment_limit = (separation_distance + alignment_distance) / radii_sector` — normalized threshold
- Position update multiplier: `velocity × Δt × 15`

### Distance

All pairwise distances use squared magnitude (avoiding sqrt):

```
d² = |otherPos − B.pos|²
```

### Normalized Rate

Each neighbor's squared distance is mapped to [0, 1] within the perception sphere:

```
rate = d² / area_sector    // rate ∈ [0, 1]
```

### Three-Zone Flocking (All-Pairs O(n²))

Every boid is compared against every other boid. The `rate` determines which of three zones applies:

#### Zone 1: Separation (rate < separation_limit)

Repulsion with asymptotic strength as distance → 0:

```
change = (separation_limit / rate − 1.0) × Δt
nextmoment_velocity −= normalize(otherPos − B.pos) × change
```

As `rate → 0`, `change → ∞` — the closer a neighbor, the stronger the push.

#### Zone 2: Alignment (separation_limit ≤ rate < alignment_limit)

Cosine-weighted matching of neighbor velocity, strongest in the middle of the zone:

```
rate_adjust = (rate − separation_limit) / (alignment_limit − separation_limit)    // ∈ [0, 1]
change = (0.5 − cos(rate_adjust × 2π) × 0.5 + 0.5) × Δt
nextmoment_velocity += normalize(neighbor.velocity) × change
```

| rate_adjust | change / Δt |
|:-----------:|:-----------:|
| 0.0 (near)  | 0.5         |
| 0.25        | 0.85        |
| 0.5 (mid)   | 1.5         |
| 0.75        | 0.85        |
| 1.0 (far)   | 0.5         |

This produces a bell-shaped influence profile — strongest alignment at the middle of the zone, fading at both edges.

#### Zone 3: Cohesion (rate ≥ alignment_limit)

Cosine-weighted steer toward neighbor position:

```
rate_adjust = (rate − alignment_limit) / (1.0 − alignment_limit)    // ∈ [0, 1]
change = (0.5 − cos(rate_adjust × 2π) × (−0.5) + 0.5) × Δt
       = (1.0 + cos(rate_adjust × 2π) × 0.5) × Δt
nextmoment_velocity += normalize(otherPos − B.pos) × change
```

| rate_adjust | change / Δt |
|:-----------:|:-----------:|
| 0.0 (near)  | 1.5         |
| 0.25        | 0.85        |
| 0.5 (mid)   | 0.5         |
| 0.75        | 0.85        |
| 1.0 (far)   | 1.5         |

Cohesion is strongest at the boundaries of the zone (near and far edges), weakest in the middle.

### Center Attraction

A constant pull toward the origin, biased against vertical flight:

```
toCenter = B.pos − (0, 0, 0) = B.pos
toCenter.y *= 2.5    // amplify Y to discourage altitude

nextmoment_velocity −= normalize(toCenter) × Δt × 5.2
```

### Predator Avoidance

A falcon position (uniform `vec3`) drives escape behavior:

```
falconDirection = falcon × upper_bound − B.pos
falconDirection.z = 0    // ignore Z, flee in XY plane only

if |falconDirection| < radii_falcon_view:
    change = (|falconDirection|² / radii_falcon_view² − 1.0) × Δt × 100
    escapeVelocity += normalize(falconDirection) × change
    limit = speed_limit + 5 = 15    // boost speed when fleeing
```

As the predator gets closer (`|falconDirection|²` decreases), `change` becomes more negative (larger in magnitude), producing stronger escape acceleration.

### Speed Limiting

```
if |nextmoment_velocity| > limit:
    nextmoment_velocity = normalize(nextmoment_velocity) × limit
```

### Position Update

```
B.pos += B.velocity × Δt × 15
```

The ×15 multiplier accelerates movement relative to the velocity computation.

### Wing Flap (4th Texture Component)

The 4th (alpha) channel of the position texture accumulates a wing-flap phase:

```
flap += Δt + |velocity.xz| × Δt × 3 + max(velocity.y, 0) × Δt × 6
flap = mod(flap, 62.83)    // ≡ mod(flap, 20π)
```

Flap speed increases with horizontal speed and climbing (positive Y).

### Initialization (Texture Fill)

```
B.pos     = (rand(0, 100), rand(0, 100), rand(0, 100))    // ∈ [0, 100]³
B.vel     = (rand(−0.5, 0.5), rand(−0.5, 0.5), rand(−0.5, 0.5)) × 10    // ∈ [−5, 5]³
B.flap    = 1
```

## 09

A Scala 2D boids simulation with GUI controls and sliders. Uses **velocity-averaging inertia** — the current velocity is included in the force summation. Obstacle avoidance uses a **"see ahead" ray** with three sample points. All rules output normalized unit vectors weighted and summed, then the velocity is normalized to `topSpeed`.

### Constants

- `topSpeed = 3` — fixed speed (velocity always normalized to this)
- `detectionRadius = 50` — perception radius
- `separationWeight = 5`, `cohesionWeight = 4`, `alignmentWeight = 6`
- `centerPull = 1` — attraction to center of screen
- `seeAhead = 90` — look-ahead distance for obstacle detection
- `avoidanceWeight = 85`
- `ruleMultiplier = 0.2` — global scaling applied to all rule forces
- `obstacleRadiusMultiplier = 1.2` — safety margin around obstacles
- `vehicleLimit = 1000`, screen: `width = 1600`, `height = 600`

### Vector Operations

```
|v|     = √(v.x² + v.y²)
v̂      = v / |v|    (if |v| ≠ 0)
v₁ + v₂ = (v₁.x + v₂.x, v₁.y + v₂.y)
v₁ − v₂ = (v₁.x − v₂.x, v₁.y − v₂.y)
v × s   = (v.x × s, v.y × s)
d(v₁, v₂) = √((v₁.x − v₂.x)² + (v₁.y − v₂.y)²)
```

### Separation

Sum of unit vectors pointing away from ALL neighbors within `detectionRadius`, then normalized:

```
for each neighbor where d(B, neighbor) < detectionRadius:
    repulsionVector += normalize(B.position − neighbor.position)

separation = normalize(repulsionVector)
```

All neighbors contribute equally regardless of distance (no 1/d falloff).

### Cohesion

Normalized vector from boid toward the center of mass of neighbors:

```
centerOfMass = (1/N) × Σ neighbor.position
cohesion = normalize(centerOfMass − B.position)
```

### Alignment

Normalized average velocity of neighbors:

```
avgVelocity = (1/N) × Σ neighbor.velocity
alignment = normalize(avgVelocity)
```

### Center Attraction

```
center = (width / 2, height / 2)
vectorToCenter = normalize(center − B.position)
```

### Obstacle Avoidance (See-Ahead Ray)

A "whisker" ray is projected ahead of the boid. Obstacles intersecting any of three sample points (ahead, half-ahead, and current position) trigger avoidance toward the closest threat:

```
ahead     = B.position + normalize(B.velocity) × seeAhead
aheadHalf = ahead × 0.5

// An obstacle is intersected if ANY of these points is within its radius:
intersects(B, obstacle) = 
    d(ahead, obstacle.pos) ≤ obstacle.radius × 1.2 OR
    d(aheadHalf, obstacle.pos) ≤ obstacle.radius × 1.2 OR
    d(B.position, obstacle.pos) ≤ obstacle.radius × 1.2

threat = argmin(d(B.position, o.pos)) over intersected obstacles

if threat exists:
    avoidance = normalize(ahead − threat.position)
else:
    avoidance = (0, 0)
```

### Velocity Inertia

The boid's current velocity is included as one of the force terms, creating directional inertia:

```
actingVectors = [
    B.velocity,                                          // inertia (weight = 1)
    separation     × ruleMultiplier × separationWeight,
    cohesion       × ruleMultiplier × cohesionWeight,
    alignment      × ruleMultiplier × alignmentWeight,
    vectorToCenter × ruleMultiplier × centerPull
]

if obstacles exist:
    actingVectors += avoidance × ruleMultiplier × avoidanceWeight

B.velocity = normalize( Σ actingVectors ) × topSpeed
```

This means: `newVelocity = normalize(velocity + sep*weight + coh*weight + ali*weight + center*weight [+ avoid*weight]) × topSpeed`. The current direction persists as a force — the boid tends to keep moving in its current direction.

### Position Update & Boundary Wrap

Wrap-around boundaries. Due to an if-else chain, wrapping and movement are mutually exclusive — a boid stops moving while being teleported:

```
if pos.x > width:
    pos.x = pos.x % width
else if pos.y > height − 195:
    pos.y = pos.y % (height − 195)
else if pos.x < 0:
    pos.x += width
else if pos.y < 0:
    pos.y += (height − 195)
else:
    position += velocity
```

### Initialization

```
B.velocity = (cos(randAngle), sin(randAngle))    // unit vector at random angle
B.position = (rand(0, width), rand(0, height))
```

## 10

A Unity GPU compute shader implementation with two variants: a **basic O(n²) all-pairs** version and an **optimized parallel reduction** version using prefix-sum for O(log n) global aggregation. Both use exponential smoothing for heading interpolation and share the same steering math.

### Constants

- `moveSpeed = 1` — movement speed
- `separationWeight = 0.5` (range [0, 1])
- `alignmentWeight = 0.5` (range [0, 1])
- `targetWeight = 0.5` (range [0, 1])
- `boidExtent = (32, 32, 32)` — spawn volume half-extents
- Thread group size: basic = 8, parallel reduction = 32

### Safe Normalize

```
normalizeSafe(v) = normalize(v) if |v| > 0, else (0, 0, 0)
```

### 10a — Basic O(n²) Variant

Every boid iterates over ALL boids (including itself, no skip-self check). Global averages are computed per-boid.

#### Global Average Computation

```
cellAlignment  = Σ allBoids.forward
cellSeparation = Σ allBoids.position

avgForward = cellAlignment / numBoids
avgPosition = cellSeparation / numBoids
```

Since self is included, `avgForward` and `avgPosition` are true global means (with negligible self-contribution for large N).

#### Alignment

Steer toward the global average forward direction:

```
alignmentResult = alignmentWeight × normalizeSafe(avgForward − B.forward)
```

#### Separation

```
separationResult = separationWeight × normalizeSafe((B.position / numBoids) − cellSeparation)
```

This approximates a vector pointing away from the global center of mass (self-inclusive formulation).

#### Target Attraction

```
targetHeading = targetWeight × normalizeSafe(targetPosition − B.position)
```

#### Heading Blending & Integration

All three forces are summed and normalized, then the heading is exponentially smoothed toward the desired direction:

```
normalHeading = normalizeSafe(alignmentResult + separationResult + targetHeading)
nextHeading   = normalizeSafe(B.forward + Δt × (normalHeading − B.forward))

B.position += nextHeading × moveSpeed × Δt
B.forward   = nextHeading
```

The term `forward + Δt × (target − forward)` is exponential smoothing — the boid turns toward `normalHeading` at a rate determined by `Δt`.

### 10b — Parallel Reduction Variant

Replaces the O(n²) per-boid loop with a two-pass GPU parallel reduction for O(log n) global aggregation.

#### Pass 1 — Parallel Reduction Sum

Uses a tree-based inclusive prefix sum within shared memory blocks of 32 threads. Multiple dispatch passes reduce the buffer hierarchically until `boidPrefixSumBuffer[0]` contains the sum of all positions and forwards.

```
// Per thread group of BLOCK_SIZE = 32:
shared[threadIdx] = boidBuffer[dispatchIdx]

for stride = 1, 2, 4, 8, 16:
    barrier()
    tmp = shared[threadIdx]
    if threadIdx ≥ stride:
        tmp.forward  += shared[threadIdx − stride].forward
        tmp.position += shared[threadIdx − stride].position
    barrier()
    shared[threadIdx] = tmp

boidPrefixSumBuffer[dispatchIdx] = shared[threadIdx]
```

After all reduction passes:

```
totalForward  = boidPrefixSumBuffer[0].forward
totalPosition = boidPrefixSumBuffer[0].position
```

#### Pass 2 — Steering with Global Averages

Each boid reads the pre-computed global averages directly:

```
globalAvgForward  = totalForward / numBoids
globalAvgPosition = totalPosition / numBoids

alignmentResult  = alignmentWeight × normalizeSafe(globalAvgForward − B.forward)
separationResult = separationWeight × normalizeSafe(B.position − globalAvgPosition)
```

Note: this variant correctly computes `B.position − globalAvgPosition` (pointing away from center of mass), unlike the basic variant.

Target attraction, heading blending, and position update are identical to 10a.

### Initialization

```
B.position = (rand(−32, 32), rand(−32, 32), rand(−32, 32))    // uniform in box
B.forward  = rotate(randomQuaternion, (0, 0, 1))               // random unit direction on sphere
```

## 11

A feature-rich Unity 3D simulation with **boids and predators**, skeletal animation, and state machines. Uses **smooth behavior blending** via distance-based inverse lerp weights, a **field-of-view cone** for neighbor perception, **dynamic vision distance adaptation**, **random walk** for isolated boids, and **8-directional raycast obstacle avoidance** with distance-weighted blending.

### Constants

- `visionDistance` — max perception distance (adaptively adjusted per boid)
- `visionSemiAngle` — half-angle of the FOV cone
- `cosVisionSemiAngle = cos(visionSemiAngle)` — precomputed FOV threshold
- `separationRadius`, `cohesionRadius`, `fearRadius` — behavioral zone radii
- `smoothnessRadiusOffset` — gradient width for smooth zone transitions
- `separationBaseWeight`, `alignmentBaseWeight`, `cohesionBaseWeight`, `fearBaseWeight` — rule weights
- `momentumWeight` — tendency to keep current direction
- `idealNbNeighbors` — target neighbor count for vision adaptation
- `acceleration`, `emergencyAcceleration` — speed change rates
- `velocityBonusFactor ∈ [minFactor, maxFactor]` — random speed multiplier
- `rwStatePeriod`, `rwMomentumWeight`, `rwProbaStraightLine`, `rwVerticalRestriction` — random walk params
- `raycastBaseDistance`, `obstacleBaseMargin` — obstacle detection
- Predator-specific: `preyAttractionBaseWeight`, `peerRepulsionBaseWeight`, state machine probabilities, wave animation params

### Field of View (FOV) Check

A boid only considers entities within a vision cone:

```
cosAngle = (entityPosition − myPosition).normalized · myDirection
if cosAngle ≥ cosVisionSemiAngle:  entity is in FOV
```

### Distance-Based Weight Functions

All behavioral weights use `InverseLerp` for smooth transitions across gradient zones:

```
visionWeight(d²)   = InverseLerp(visionDistance², (visionDistance − 1)², d²)
```

This is 1 when close, 0 at vision limit, with smooth fade in the outermost unit.

For the three behavioral zones, each neighbor contributes a blended weight split across separation, alignment, and cohesion. The split is based on where the neighbor falls within the two threshold radii:

```
separationPortion(d²) = InverseLerp(sepRadius², fullSepRadius², d²)      // 1 when close → 0
cohesionPortion(d²)    = InverseLerp(cohRadius², fullCohRadius², d²)     // 0 → 1 when far
alignmentPortion(d²)   = 1 − separationPortion − cohesionPortion

fearWeight(d²)         = InverseLerp(fearRadius², fullFearRadius², d²)   // for predators
```

Where `fullSepRadius² = (sepRadius − smoothOffset)²` (smaller, inner bound) and `fullCohRadius² = (cohRadius + smoothOffset)²` (larger, outer bound).

### Per-Neighbor Weighted Contributions

For each neighbor boid in FOV:

```
w           = visionWeight(d²)
wSep        = w × separationPortion(d²)
wAli        = w × alignmentPortion(d²)
wCoh        = w × cohesionPortion(d²)

weightedNbSep    += wSep
weightedNbAli    += wAli
weightedNbCoh    += wCoh

sepPosSum        += wSep × neighbor.position
cohPosSum        += wCoh × neighbor.position
aliDirSum        += wAli × neighbor.direction
```

For each predator:

```
wFear = fearWeight(d²)
weightedNbFear += wFear
fearPosSum     += wFear × predator.position
```

### Ideal Behavior Directions

```
alignmentDir  = normalize(aliDirSum)
cohesionDir   = normalize(cohPosSum / weightedNbCoh − myPosition)
separationDir = −normalize(sepPosSum / weightedNbSep − myPosition)  // opposite
fearDir       = −normalize(fearPosSum / weightedNbFear − myPosition) // flee
```

### Real Weight Scaling

Each behavior's weight saturates at 1 to avoid over-domination by many neighbors:

```
realWeight = min(weightedNbForBehavior, 1) × baseWeight
```

### Final Direction Blend (Boid)

```
newDirection = normalize(
    momentumWeight × currentDirection +
    realSepWeight × separationDir +
    realAliWeight × alignmentDir +
    realCohWeight × cohesionDir +
    realFearWeight × fearDir
)
```

Momentum (the current direction as a force term) creates directional inertia.

### State Machine

```
if predators nearby:               state = AFRAID
else if nbBoidsNearby < threshold:  state = ALONE
else:                               state = NORMAL
```

- **ALONE**: uses Random Walk instead of flocking rules
- **AFRAID**: fear direction included, emergency acceleration applied
- **NORMAL**: standard flocking

### Visual Distance Adaptation

Each boid dynamically adjusts its vision distance to maintain approximately `idealNbNeighbors` in its FOV:

```
if nbInFOV > idealNbNeighbors:  visionDistance = max(1, visionDistance − 1)
if nbInFOV < idealNbNeighbors:  visionDistance = min(maxVision, visionDistance + 1)
```

### Random Walk

For ALONE boids, chains of straight-line and direction-change segments:

```
if Bernoulli(probStraightLine):   continue straight
else:
    targetDir = normalize(randomDir + currentDir × rwMomentumWeight)
    // reject if raycast hits obstacle, retry up to maxAttempts

// Smooth interpolation within segment:
progress = (statePeriod − timeRemaining) / statePeriod
newDir = normalize(Lerp(lastDir, targetDir, progress))
```

### Obstacle Avoidance (8-Directional Raycast)

When a forward raycast hits an obstacle:

1. Generate 8 avoidance directions perpendicular to the heading, using `(axis1, axis2)` orthogonal to `direction`:
   ```
   dirs = [axis1, −axis1, (axis1+axis2)/√2, (axis1−axis2)/√2,
           (−axis1+axis2)/√2, (−axis1−axis2)/√2, axis2, −axis2]
   ```
   For boids, `(axis1, axis2)` are generated with random reference → no bias. For predators, reference is `Vector3.up` → horizontal avoidance is preferred.

2. For each avoidance direction, blend with original direction based on obstacle distance:
   ```
   perceivedDist = Remap(hitDist, margin, raycastDist, 0, raycastDist)
   blendedDir = normalize(originalDir × perceivedDist + avoidDir × (raycastDist − perceivedDist))
   ```
   When obstacle is far: `originalDir` dominates. When close: `avoidDir` dominates.

3. Raycast in blended direction. If clear → use it. If hit → weight by preference and choose best.

### Direction Smoothing (Spherical Interpolation)

```
progress = sinceLastCalc / calculationInterval
newRotation = Slerp(lastRotation, targetRotation, progress)
direction   = newRotation × Vector3.forward
```

### Movement

```
newPosition = myPosition + velocity × Δt × direction
```

### Velocity Adaptation

Velocity smoothly approaches a state-dependent target speed with a random bonus factor:

```
goal    = baseVelocity[state] × randomBonusFactor
accel   = emergencyAcceleration if AFRAID/ATTACKING, else acceleration
step    = accel × Δt
velocity → lerp toward goal at rate step

// Bonus factor cycles: every period, randomFactor = Random(minFactor, maxFactor)
```

### Predator Behavior

Predators use a simpler two-behavior blend plus a state machine:

```
preyAttractionDir = normalize(preyPosSum / weightedNbPrey − myPosition)
peerRepulsionDir  = −normalize(peerPosSum / weightedNbPeers − myPosition)

newDirection = normalize(
    momentumWeight × currentDir +
    preyWeight × preyAttractionDir +
    peerWeight × peerRepulsionDir
)

// Restrict vertical angle:
newDirection = RestrictDirectionVertically(newDirection, verticalRestriction)
```

State machine (probabilistic):
```
CHILLING → HUNTING:  Bernoulli(pHuntingAfterChilling)
HUNTING  → ATTACKING: nbPreyInFOV > nbPreysToAttack
HUNTING  → CHILLING:  Bernoulli(pChillingAfterHunting)
ATTACKING → CHILLING or HUNTING: Bernoulli(pHuntingAfterAttacking)
```

### Predator Wave Animation

Sinusoidal lateral displacement of bones along the body:

```
velocityFactor = 1 + velocityImpact × (velocity / baseVelocity − 1)
speed = velocityFactor × baseWaveSpeed
magnitude = velocityFactor × baseWaveMagnitude
phase += speed (mod 2π)

for each bone at distance d from head:
    displacement = magnitude × (envelopeGradient × d + envelopeMin)
                   × sin(spatialFreq × d − phase)
    bonePos += displacement × right
```

### Skeletal Trail Animation (Boids)

Bones follow the historical trajectory — each bone occupies a position the body passed through earlier:

```
for each bone at distance d from head:
    step back through trajectory history until traveled distance ≥ d
    interpolate between two history frames to find exact position/rotation
    boneTransform = Lerp/Slerp(framePos, prevFramePos, fractionalStep)
```

### Spherical-to-Cartesian Conversion

Used for random direction generation and direction restriction:

```
// Spherical → Cartesian:
x = sin(φ) × cos(θ)
y = cos(φ)
z = sin(φ) × sin(θ)

point = center + distance × (x, y, z)
```

### Initialization

```
B.direction = randomDirection(verticalRestriction)   // random on sphere, optionally flattened
B.position  = random(aabb.min, aabb.max)             // uniform in spawn area
B.velocity  = baseVelocity[defaultState]
B.visionDist = visionDistance
```

## 12

A Unity compute shader sandbox with both **CPU (C#)** and **GPU (HLSL)** implementations sharing the same Reynolds steering math. Uses proper `desired − velocity` steering vectors. The GPU variant adds a **1/d wall repulsion force**. Both O(n²) per boid.

### Constants

- `boidsCount = 50` — number of boids
- `insightRange = 3` (aka `effectRange`) — perception radius
- `maxVelocity = 0.1` — speed limit
- `maxAcceleration = 0.1` — acceleration/force limit
- `fleeThreshold = 1` — separation distance trigger
- `alignWeight = 1`, `separationWeight = 1`, `cohesionWeight = 1`
- `wallForceWeight`, `wallDistanceWeight` — wall force params (GPU only)
- `boundarySize` — box half-extents, e.g. (5, 5, 5)

### Distance (Squared)

```
d² = dot(otherPos − selfPos, otherPos − selfPos)
withinRange(d² < range²)
```

### Vector Limiting

```
// GPU (correct):
LimitVector(v, maxLen) = normalize(v) × maxLen    if |v| ≠ 0, else (0,0,0)

// C# (buggy — divides by sqrMagnitude instead of magnitude):
LimitVector(v, maxLen) = v × (maxLen / |v|²)      if |v|² > maxLen²
```

### Alignment

Reynolds steering toward average neighbor velocity:

```
avgVel = (1/N) × Σ neighbor.velocity    for neighbors within insightRange
alignForce = avgVel − self.velocity
```

### Separation

Steer away from neighbors closer than `fleeThreshold`. Uses an unconventional formulation incorporating own velocity:

```
for neighbors within insightRange:
    displacement = neighbor.position − self.position
    if |displacement|² ≥ fleeThreshold²: skip
    fleeForce += −(displacement − self.velocity)
```

Where `−(displacement − velocity) = velocity − displacement`. The inclusion of `−self.velocity` adds a velocity-dependent term to the avoidance.

### Cohesion

Reynolds steering toward average neighbor position:

```
avgPos    = (1/N) × Σ neighbor.position    for neighbors within insightRange
desired   = avgPos − self.position
seekForce = desired − self.velocity
```

### Force Combination

```
force = alignWeight × alignForce
      + separationWeight × separationForce
      + cohesionWeight × cohesionForce

force = LimitVector(force, maxAcceleration)
```

### Velocity & Position Integration

```
velocity += force × Δt
velocity  = LimitVector(velocity, maxVelocity)
position += velocity × Δt
```

### Border Treatment (Reflective)

Hard box boundary with velocity mirroring:

```
if pos.x > +boundary.x:  pos.x = +boundary.x;  vel.x = −vel.x
if pos.x < −boundary.x:  pos.x = −boundary.x;  vel.x = −vel.x
// same for y, z
```

### Wall Force (GPU Only)

A 1/d repulsion from each of the 6 wall faces, applied after force clamping:

```
wallForce = (−1,0,0) / |(+boundary.x − pos.x) / distWeight|
          + (+1,0,0) / |(−boundary.x − pos.x) / distWeight|
          + (0,−1,0) / |(+boundary.y − pos.y) / distWeight|
          + (0,+1,0) / |(−boundary.y − pos.y) / distWeight|
          + (0,0,−1) / |(+boundary.z − pos.z) / distWeight|
          + (0,0,+1) / |(−boundary.z − pos.z) / distWeight|

force += wallForceWeight × wallForce
```

Each wall pushes inward with strength inversely proportional to the normalized distance. When the boid approaches a wall, the denominator shrinks and the repulsion grows. In the GPU path, this replaces `BorderTreatment` (which is commented out).

### Orientation

```
rotation = LookRotation(velocity)
```

### Initialization

```
B.position = Random(−boundarySize, +boundarySize)    // per-component uniform
B.velocity = Random(−maxVelocity, +maxVelocity)       // per-component uniform
```

## 13

A Blazor WebAssembly (C# / .NET) 2D boids simulation with HTML Canvas rendering. Uses standard Reynolds steering for alignment and cohesion. Features **exponential separation** (strongest at zero distance, decaying as `exp(−(d−sepDist)/sepDist)`), **1/d weighted cohesion**, and **linear wall avoidance**. Includes random direction jitter.

### Constants

- `Count = 100` — number of boids
- `MaxSpeed = 9` — speed limit
- `PerceptionRadius = 50` — neighbor perception radius
- `SeparationDistance = 20` — reference distance for separation
- `DirectionChangeFactor = 0.5` — random jitter strength
- `AvoidWallsDistance = 50` — distance from edge to begin steering
- `Fps = 30` — target frame rate

### Euclidean Distance

```
d = |other.Position − self.Position|
```

Distance is computed once during perception gathering and stored on each neighbor object for reuse.

### Perception

```
for each other boid:
    d = |other.Position − self.Position|
    if d ≤ PerceptionRadius and other ≠ self:
        include in perception list
```

### Random Direction Change

30% chance per frame of adding random jitter:

```
if rand < 0.7: skip    // 70% chance no change
newVel = randomDirection × MaxSpeed
Velocity += newVel × DirectionChangeFactor
```

### Alignment

Standard Reynolds: desired velocity is average neighbor velocity scaled to `MaxSpeed`:

```
avgVel  = (1/N) × Σ neighbor.Velocity
desired = normalize(avgVel) × MaxSpeed
alignForce = desired − Velocity
```

### Cohesion (Weighted — active variant)

Weighted center of mass using 1/d weighting, then Reynolds steering:

```
for each neighbor:
    weight = 1 / d
    weightedPosSum += neighbor.Position × weight
    totalWeight    += weight

weightedCOM  = weightedPosSum / totalWeight
desired      = normalize(weightedCOM − Position) × MaxSpeed
cohForce     = desired − Velocity
```

Alternative (COMCohesion): unweighted center of mass, otherwise identical.

### Separation (Exponential — active variant)

Exponentially decaying repulsion, strongest at zero distance:

```
combinedRadius = SeparationDistance × 2    // = 40

for each neighbor where d < combinedRadius:
    r̂ = (Position − neighbor.Position) / d         // unit vector away
    factor = exp(−(d − SeparationDistance) / SeparationDistance)
    separationForce += r̂ × factor
```

| d       | factor ≈ |
|:-------:|:--------:|
| 0       | e¹ ≈ 2.72 |
| sepDist (20) | e⁰ = 1.00 |
| 30      | e⁻⁰·⁵ ≈ 0.61 |
| 40      | e⁻¹ ≈ 0.37 |
| > 40    | 0 (excluded) |

Alternative (NormalizedWeightingSeparate): `avg(r̂)` per neighbor, then Reynolds steering.

### Force Combination

```
Velocity += alignForce + cohesionForce + separationForce
```

### Speed Limiting

Proportional scaling (preserves direction):

```
speed = |Velocity|
if speed > MaxSpeed:
    Velocity *= MaxSpeed / speed
```

### Wall Avoidance

Linear ramp from edge to `AvoidWallsDistance`:

```
if pos.x < avoidDist:
    steer.x += (avoidDist − pos.x) / avoidDist
else if pos.x > width − avoidDist:
    steer.x −= (pos.x − (width − avoidDist)) / avoidDist

// same for y

Velocity += steer
```

At the wall: `steer = 1`. At distance `avoidDist`: `steer = 0`. Linear in between.

### Position Update

```
Position += Velocity    // no Δt — velocity is per-frame
```

### Initialization

```
B.Position = (rand(0, width), rand(0, height))
B.Velocity = (rand(−1, 1), rand(−1, 1)) × MaxSpeed
```

## 14

A Unity ECS project with multiple framework samples (Pure ECS, LeoECS, SveltoECS, MonoBehaviour). All share the same core math via a common `Param` ScriptableObject. Uses **FOV-based neighbor detection** with a cosine cone test, **direct position-difference cohesion** (no Reynolds steering), and a **1/d wall repulsion** force.

### Constants

- `initSpeed = 2` — initial speed
- `minSpeed = 2` — minimum speed (clamped lower bound)
- `maxSpeed = 5` — maximum speed (clamped upper bound)
- `neighborDistance = 1` — perception radius
- `neighborFov = 90°` — field of view half-angle (full 180° cone)
- `separationWeight = 5`
- `alignmentWeight = 2`
- `cohesionWeight = 3`
- `wallScale = 5` — bounding box full size
- `wallDistance = 3` — wall avoidance activation distance
- `wallWeight = 1`

### FOV-Based Neighbor Detection

Neighbors must be within both distance threshold AND forward-facing hemisphere:

```
prodThresh = cos(neighborFov)    // cos(90°) = 0

d = |other.pos − self.pos|
if d < neighborDistance:
    d̂ = normalize(other.pos − self.pos)
    f̂ = normalize(self.velocity)
    if dot(f̂, d̂) > prodThresh:       // within FOV cone
        add to neighbors
```

With `neighborFov = 90°`, the threshold is `cos(90°) = 0`, meaning any neighbor in the forward 180° hemisphere is included.

### Separation

Average of unit repulsion vectors from all neighbors:

```
force = (1/N) × Σ normalize(self.pos − neighbor.pos)
accel += force × separationWeight
```

All neighbors contribute equally regardless of distance.

### Alignment

Standard Reynolds steering toward average velocity:

```
avgVel = (1/N) × Σ neighbor.velocity
accel += (avgVel − self.velocity) × alignmentWeight
```

### Cohesion

Direct position-difference (no normalization, no Reynolds steering):

```
avgPos = (1/N) × Σ neighbor.pos
accel += (avgPos − self.pos) × cohesionWeight
```

The raw `(centerOfMass − pos)` vector is used — magnitude naturally scales with distance from the group.

### Wall Force

Six-face 1/d repulsion from a box boundary of half-extent `wallScale × 0.5`:

```
for each wall face (right, up, forward, left, down, back):
    dist = faceCoord − pos.component
    if dist < wallDistance:
        accel += direction × (wallWeight / |dist / wallDistance|)
```

Expanding: `accel += direction × wallWeight × wallDistance / |dist|`. As `dist → 0`, force → ∞.

| dist       | force multiplier |
|:----------:|:----------------:|
| wallDistance | wallWeight (1) |
| wallDist/2 | 2               |
| wallDist/4 | 4               |
| → 0        | → ∞             |

### Movement Integration

```
velocity += accel × Δt
dir       = normalize(velocity)
speed     = |velocity|
velocity  = clamp(speed, minSpeed, maxSpeed) × dir    // both min and max enforced
position += velocity × Δt
rotation  = LookRotation(velocity)
accel     = (0, 0, 0)    // reset each frame
```

Speed is **clamped** to [minSpeed, maxSpeed] via `clamp(|v|, minSpeed, maxSpeed) × normalize(v)`. Boids cannot stop.

### Initialization

```
B.position = Random.insideUnitSphere               // random in unit sphere
B.rotation = Random.rotation                        // random orientation
B.velocity = forward × initSpeed                    // direction × 2
```

## 15

A Unity DOTS (ECS) + Burst simulation with spatial partitioning, priority-ordered force accumulation, wander behavior, obstacle raycast avoidance, and physics-informed movement integration (smoothed acceleration, banking, velocity damping).

### Constants

- `separationWeight = 1`, `cohesionWeight = 2`, `alignmentWeight = 1`, `wanderWeight = 1`
- `constrainWeight = 1`, `fleeWeight = 1`, `fleeDistance = 50`
- `neighbourDistance = 20` — perception radius
- `totalNeighbours = 50` — max neighbors tracked
- `radius = 2000` — spherical constrain radius
- Per-boid: `mass = 1`, `maxSpeed = 100`, `maxForce = 400`, `weight = 200`
- `damping = 0.01`, `banking = 0.01` — integration parameters
- `limitUpAndDown = 0.5` — vertical acceleration dampening
- Obstacle: `forwardFeelerDepth = 50`
- Wander: `distance = 2` (ahead), `radius = 1.2` (circle), `jitter = 80`
- Optional spatial partitioning: `cellSize = 50`

### Neighbor Detection

O(n²) or spatial-partitioned (grid cells). Both use distance threshold:

```
if |neighborPos − myPos| < neighbourDistance: add to neighbor list
```

The spatial partition maps `(x, z) → cell = floor(x/cellSize) + floor(z/cellSize) × gridSize` (optionally 3D with Y). Neighboring cells are searched up to `ceil(neighbourDistance / cellSize)` steps away.

### Separation

1/d repulsion from each neighbor:

```
for each neighbor:
    toNeighbor = myPos − neighborPos
    mag = |toNeighbor|
    if mag > 0:
        force += normalize(toNeighbor) / mag      // r̂ / d
    else:
        force += randomDirection × maxForce       // fallback for coincident positions

separation.force = force × separationWeight
```

### Alignment

Standard Reynolds steering toward average forward:

```
avgForward = (1/N) × Σ neighbor.forward     (forward = rotation × (0,0,1))
force = avgForward − myForward
alignment.force = force × alignmentWeight
```

### Cohesion

Reynolds seek toward center of mass:

```
centerOfMass = (1/N) × Σ neighborPos
toTarget     = centerOfMass − myPos
desired      = normalize(toTarget) × maxSpeed
force        = normalize(desired − velocity)

cohesion.force = force × cohesionWeight
```

### Wander

Classic Reynolds wander: a point jitters randomly on a circle projected ahead of the boid:

```
target += jitter × randomDirection × Δt
target  = normalize(target) × wanderRadius

localTarget  = (0, 0, wanderAheadDistance) + target     // ahead + circle point
worldTarget  = rotation × localTarget + position

wander.force = (worldTarget − position) × wanderWeight
```

### Spherical Constrain

Linear spring force pushing boids back inside a sphere:

```
toCenter = position − sphereCenter
if |toCenter| > sphereRadius:
    force = normalize(toCenter) × (sphereRadius − |toCenter|)

constrain.force = force × constrainWeight
```

Force magnitude is proportional to how far outside the boid is (`radius − distance`).

### Obstacle Avoidance (Raycast)

Single forward raycast using Unity Physics:

```
forward = normalize(rotation × (0,0,1))
raycast from position to position + forward × forwardFeelerDepth

if hit:
    dist = distance(hit.point, position)
    force = hit.normal × (forwardFeelerDepth / dist)

obstacle.force = force
```

The `feelerDepth / dist` term means the avoidance force grows stronger as the obstacle gets closer.

### Priority-Ordered Force Accumulation

Forces are accumulated in a strict priority order. After each addition, the total force is clamped to `maxForce` — giving earlier forces higher priority:

```
force = 0

force += obstacle.force;     clamp(force, maxForce); if at cap: return
force += fleeForce;           clamp(force, maxForce); if at cap: return
force += separation.force;    clamp(force, maxForce); if at cap: return
force += alignment.force;     clamp(force, maxForce); if at cap: return
force += cohesion.force;      clamp(force, maxForce); if at cap: return
force += wander.force;        clamp(force, maxForce); if at cap: return
force += constrain.force;     clamp(force, maxForce); if at cap: return
force += seekForce;           clamp(force, maxForce); if at cap: return
```

Priority: obstacle > flee > separation > alignment > cohesion > wander > constrain > seek.

### Movement Integration

Physically-informed with smoothed acceleration, banking, and velocity damping:

```
force  = accumulatedForces × boidWeight
force  = clamp(force, maxForce)

newAccel = (force × boidWeight) × (1 / mass)     // F = m·a → a = F/m
newAccel.y *= limitUpAndDown                      // dampen vertical response

accel = Lerp(accel, newAccel, Δt)                 // smooth acceleration

velocity += accel × Δt
velocity  = clamp(velocity, maxSpeed)

if |velocity| > 0:
    // Banking: tilt the up vector toward the direction of acceleration
    tempUp = Lerp(up, worldUp + accel × banking, Δt × 3)
    rotation = LookRotation(velocity, tempUp)
    up = rotation × worldUp
    
    position += velocity × Δt
    velocity  *= (1.0 − damping × Δt)     // slight friction
```

### Initialization

```
B.position  = random.insideUnitSphere × sphereRadius + center
B.rotation  = Euler(rand(−20, 20), rand(0, 360), 0)
B.maxSpeed  = 100 × rand(0.9, 1.1)
B.mass      = 1
B.maxForce  = 400
B.weight    = 200
```

## 16

A Rust + Nannou 2D simulation using **angle-based movement** (not vector-based). Separation and cohesion work identically — both compute a target angle and rotate the bird toward/away from it by a fixed delta per frame. Alignment uses a **circular mean** of neighbor angles applied as a gain. An **edge state machine** with three intensity levels handles boundaries via exponential flocking attenuation near edges.

### Constants

- `BIRD_REGION_RADIUS = 225.0` — perception radius
- `BIRD_SEPARATION_RADIUS = 30.0` — separation radius
- `EDGE_BLEED = 50.0` — screen-wrap buffer
- `TURN_GAIN = 0.020` — edge turning rate
- `HARD_ANGLE_MULTIPLIER = 5.0` — aggressive turn multiplier
- `HARD_ANGLE_SATURATION = 65°` — max turn rate in harder mode
- `DISTANCE_DECAY = 0.1` — exponential decay for edge attenuation
- Separation delta: +1° per frame (turn away)
- Cohesion delta: −1° per frame (turn toward)
- Random speed per boid via `Speed` (min, max, randomise)

### Euclidean Distance

```
d = √((other.x − self.x)² + (other.y − self.y)²)
inside = d ≤ perception_radius
```

### Circular Mean of Angles

Angles are averaged correctly on the circle using `atan2` of the averaged sine/cosine:

```
avgSin = (1/N) × Σ sin(angle_i)
avgCos = (1/N) × Σ cos(angle_i)
avgAngle = atan2(avgSin, avgCos)    // ∈ [−π, π]
```

### Angle Wrapping

```
wrap(θ)    = θ mod 2π, result always in [0, 2π)
wrap_180(θ) = wrap to [−π, π]
angleDelta(a, b) = wrap_180(a − b)     // shortest angular distance
```

### Separation

The angle pointing FROM the cluster center TO the bird (i.e., away from the group):

```
avgPos  = (1/N) × Σ neighbor.position
angle   = atan2(self.y − avgPos.y, self.x − avgPos.x)    // direction away from cluster
avgAngle = circular_mean(neighbor.angles)

return (angle, avgAngle)
```

### Cohesion

The angle pointing FROM the bird TOWARD the cluster center:

```
avgPos  = (1/N) × Σ neighbor.position
angle   = atan2(avgPos.y − self.y, avgPos.x − self.x)     // direction toward cluster
avgAngle = circular_mean(neighbor.angles)

return (angle, avgAngle)
```

### Alignment

Returns the shortest angular difference between the circular mean of neighbor angles and the bird's own angle:

```
avgAngle = circular_mean(neighbor.angles)
delta    = wrap_180(avgAngle − self.angle)
return delta
```

### apply_proximity (Shared by Separation & Cohesion)

Both behaviors use the same mechanism — the only difference is the sign of `delta` (+1° for separation, −1° for cohesion):

```
/* 1. Move toward the proximity angle at half speed */
if Idle: move(direction = proxAngle, distance = speed/2)

/* 2. Compute rotation: rotate position by −alignment offset */
angle_offset = 0 − alignment    // negative of the alignment delta
rotated_pos  = rotate(old_position, angle_offset)    // 2D rotation matrix
norm_angle   = wrap(self.angle − alignment)

/* 3. Determine turn direction based on which quadrant rotated_pos falls in */
if rotated_pos.y > 0:
    if norm_angle ∈ (90°, 270°): δ = −rot_angle
    else:                         δ = +rot_angle
else:
    if norm_angle ∈ (90°, 270°): δ = +rot_angle
    else:                         δ = −rot_angle

/* 4. Apply the rotation and move forward at half speed */
self.angle = wrap(self.angle + δ)
move(direction = self.angle, distance = speed/2)
```

For separation (rot_angle = +1°): the bird turns away from neighbors.
For cohesion (rot_angle = −1°): the bird turns toward neighbors.

### Alignment Integration

```
align_gain = config.alignment_gain

if near_edge:
    dist = distance_outside(inner_bounds)
    reduct = exp(dist × −0.1)
    separation.attenuate_angle(reduct)    // *= reduct
    cohesion.attenuate_angle(reduct)      // *= reduct
    align_gain *= reduct

if separation changed: apply_proximity(separation)
if cohesion changed:   apply_proximity(cohesion)

self.angle = wrap(self.angle + align_angle × align_gain)
```

### Movement

```
x += movement_increment × cos(angle)
y += movement_increment × sin(angle)

movement_increment = random_range(minSpeed, maxSpeed)    // if randomise enabled
```

### 2D Rotation Matrix

```
x′ = x × cos(θ) − y × sin(θ)
y′ = x × sin(θ) + y × cos(θ)
```

### Edge State Machine

Three nested rectangular boundaries define turning intensity. When outside `inner`, flocking behaviors are exponentially attenuated and a state machine takes over:

```
if outside inner horizontally: → State::TurningH
if outside inner vertically:   → State::TurningV
```

**Turn angle calculation** (for right edge):
```
to_opposite = wrap(π − self.angle)        // angle needed to face opposite direction
turn_angle  = to_opposite × TURN_GAIN      // = to_opposite × 0.02
```

**State::TurningH / TurningV:**
```
self.angle += turn_angle
move at half speed
if no longer outside inner → Idle
if now outside inner_hard → TurningHarderH/V
```

**State::TurningHarderH / TurningHarderV:**
```
self.angle += saturate(turn_angle × 5.0, ±65°)
move at full speed
if no longer outside inner_hard → Idle
```

The saturation cap of 65° prevents wild spinning at the hard boundary.

### Screen Wrap

```
if x > right + EDGE_BLEED:  x = x − (width + EDGE_BLEED)
if x < left − EDGE_BLEED:   x = x + (width + EDGE_BLEED)
if y > top + EDGE_BLEED:    y = y − (height + EDGE_BLEED)
if y < bottom − EDGE_BLEED: y = y + (height + EDGE_BLEED)
```

### Initialization

```
B.position = random position
B.angle    = random angle ∈ [0, 2π)
```

## 17

A collection of Python + Pygame 2D boids simulations using **angle-based movement** with `atan2` circular mean for neighbor averaging. Four variants share identical core math: the original (`pynboids.py`), a numpy-optimized version (`pynboids2.py`), a spatial-partitioned version (`pynboids_sp.py`), a desktop-overlay variant, and a pixel-trail variant (`pixelboids.py`).

### Constants

- `BOIDZ = 88–200` — number of boids (varies by variant)
- `SPEED = 150–170` — base movement speed
- `margin = 42–48` — edge avoidance distance
- `turnRate = 120 × Δt` — steering turn speed
- `bSize = 17–22` (or `pSpace`) — boid size, used to scale behavioral radii
- Perception radius: `bSize × 12` (≈200–264) or fixed 48 (pixelboids)
- **Always keeps at most 7 closest neighbors**
- Flocking requires ≥ 2 neighbors

### Euclidean Distance

```
d² = (x₁ − x₂)² + (y₁ − y₂)²
```

### Neighbor Selection

```
1. Compute d² to all other boids
2. Sort by distance, keep 7 closest
3. Filter to only those within perception radius
4. If ≥ 2 neighbors remain: apply flocking rules
```

### Circular Mean of Angles

Neighbor angles are averaged correctly on the circle:

```
yat = Σ sin(radians(neighbor.angle))
xat = Σ cos(radians(neighbor.angle))
avgAngle = degrees(atan2(yat, xat))    // circular mean, ∈ (−180°, 180°]
```

### Target Position (Center of Mass)

```
targetV = (mean(neighbors.x), mean(neighbors.y))
```

### Three-Zone Behavior System

All three flocking rules are implemented through a single target-position/angle mechanism with distance-based overrides:

**Zone 1 — Separation** (closest neighbor < bSize):
```
if distance_to(nearestBoid) < bSize:
    targetV = nearestBoid.position    // override target to nearest neighbor
```

**Zone 2 — Alignment** (target distance < bSize × 6):
```
tDiff = targetV − self.position
tDistance, tAngle = as_polar(tDiff)

if tDistance < bSize × 6:
    tAngle = avgAngle    // override to circular mean angle
```

**Zone 3 — Cohesion** (target distance ≥ bSize × 6):
```
// tAngle stays as the angle toward the center of mass
```

### Steering Angle Computation

Convert target angle to shortest signed turn direction:

```
angleDiff = (tAngle − self.angle) + 180
turnDir = (angleDiff / 360 − floor(angleDiff / 360)) × 360 − 180
```

This is equivalent to: `turnDir = wrap_180(tAngle − self.angle)`, giving the shortest path in (−180°, 180°]. Positive = turn right, negative = turn left.

### Separation Steering Flip

When targeting the nearest boid AND too close:

```
if tDistance < bSize and targetV == nearestBoid.position:
    turnDir = −turnDir    // steer away instead of toward
```

### Edge Avoidance

When within `margin` of any screen edge, flocking is overridden by edge steering:

```
if min(x, y, maxW − x, maxH − y) < margin:
    // Set tAngle to point away from nearest edge:
    if x < margin:             tAngle = 0°     // face right
    if x > maxW − margin:      tAngle = 180°   // face left
    if y < margin:             tAngle = 90°    // face down
    if y > maxH − margin:      tAngle = 270°   // face up
    
    turnDir = wrap_180(tAngle − self.angle)
    edgeDist = min(x, y, maxW − x, maxH − y)
    turnRate = turnRate + (1 − edgeDist / margin) × (20 − turnRate)
```

The turn rate interpolates linearly from the base rate (≈120×Δt) up to 20 as distance to edge → 0.

### Angle Update

```
if turnDir ≠ 0:
    self.angle += turnRate × sign(turnDir)
    self.angle %= 360
```

### Direction & Movement

```
direction = rotate((1, 0), self.angle).normalize()    // unit vector at current angle

speed = baseSpeed + (7 − neighborCount) × bonus
position += direction × Δt × speed
```

The `(7 − neighborCount)` term makes isolated boids fly faster — up to `baseSpeed + 7 × bonus` when alone, and `baseSpeed` when surrounded by 7+ neighbors.

| Variant      | speed bonus formula     |
|-------------|------------------------|
| pynboids2   | `speed + (7-n)×2`     |
| pynboids    | `180 + (7-n)²`        |
| pynboids_sp | `speed + (7-n)×5`     |
| pixelboids  | `speed + (7-n)/14`    |

### Screen Wrap (Optional)

```
if WRAP and outside screen:
    if bottom < 0: pos.y = maxH
    if top > maxH:  pos.y = 0
    if right < 0:   pos.x = maxW
    if left > maxW: pos.x = 0
```

### Spatial Partitioning (sp/desktop variants)

Grid cells of 100×100 pixels. Boids register/unregister as they cross cell boundaries:

```
cell = (pos.x // 100, pos.y // 100)
neighbors = boids in 3×3 surrounding cells
```

### Pixel Trail Fading (pixelboids)

```
img_array[img_array > 0] −= FADE × (60/FPS/1.5) × ((Δt/10) × FPS)
img_array = clip(img_array, 0, 255)
```

### Initialization

```
B.position  = (randint(50, maxW−50), randint(50, maxH−50))
B.angle     = randint(0, 360)
B.direction = rotate((1,0), angle).normalize()
```

## 18

A Rust + Bevy 3D boids simulation with Rapier physics for raycast obstacle avoidance, golden-spiral direction sampling, a moving target, and cage boundary forces. All three flocking rules use proper Reynolds steering: `normalize(avg) × maxSpeed − velocity`, then clamped to a per-rule max force.

### Constants

- `count = 500` — number of boids
- `separation_distance = 1.0`, `alignment_distance = 1.0`, `cohesion_distance = 1.5`
- `separation_force = 0.08`, `alignment_force = 0.06`, `cohesion_force = 0.02`
- `max_speed = 4.0`
- `boundary_distance = 0.5` — cage edge activation distance
- `boundary_force_strength = 0.5` — constant cage push
- `CAGE_SIZE = 20.0` (half = 10)
- Fleet force clamp: `0.09`
- Raycast length: `4.0`
- Ray directions: 1500 points via golden spiral on sphere, 270° FOV

### Euclidean Distance

```
d = |other.position − self.position|
```

### Separation

1/d repulsion summed, averaged, then Reynolds steering:

```
for each other boid where 0 < d < separation_distance:
    r̂ = normalize(self.pos − other.pos)
    separation += r̂ / d

if neighbor_count > 0:
    separation /= neighbor_count                      // average
    separation = normalize(separation) × maxSpeed     // desired velocity
    separation = separation − velocity                // Reynolds steer
    separation = clamp(separation, separation_force)  // limit to max force
```

### Alignment

Average neighbor velocity, Reynolds steering:

```
for each other boid where d < alignment_distance:
    alignment += other.velocity

if neighbor_count > 0:
    alignment /= neighbor_count
    alignment = normalize(alignment) × maxSpeed − velocity
    alignment = clamp(alignment, alignment_force)
```

### Cohesion

Center of mass, Reynolds seek:

```
for each other boid where d < cohesion_distance:
    cohesion += other.position

if neighbor_count > 0:
    cohesion /= neighbor_count                        // center of mass
    cohesion = cohesion − self.position               // vector toward COM
    cohesion = normalize(cohesion) × maxSpeed − velocity
    cohesion = clamp(cohesion, cohesion_force)
```

Note: All three rules share a single `neighbor_count` counter that increments for any rule match, not per-rule. All three forces are divided by this common count.

### Fleet Force (Target Attraction)

Reynolds seek toward a moving target position:

```
fleet_force = target.position − self.position
fleet_force = normalize(fleet_force) × maxSpeed − velocity
fleet_force = clamp(fleet_force, 0.09)
```

### Boundary Force (Cage)

Constant push away from cage walls when close:

```
half = CAGE_SIZE / 2 = 10

if x > +half − boundaryDist:  boundary_force.x −= 0.5
if x < −half + boundaryDist:  boundary_force.x += 0.5
// same for y, z
```

### Raycast Obstacle Avoidance

Forward raycast along velocity direction. If obstructed, sample pre-generated directions to find a clear path:

```
hit = rapier.cast_ray(position, velocity, 4.0, true, only_fixed)

if hit:
    free_dir = unobstructed_dir(rapier, directions, position, velocity)
    separation += free_dir
```

#### Golden Spiral Direction Generation

1500 directions uniformly distributed on a sphere using the golden angle method, filtered to a 270° forward FOV:

```
φ = (1 + √5) / 2                            // golden ratio
angle_inc = 2π × φ
cos_threshold = cos(FOV / 2) = cos(135°) ≈ −0.707    // 270° FOV

for i in 0..1500:
    t = i / 1500
    inclination = acos(1 − 2t)              // uniform distribution on sphere
    azimuth = angle_inc × i
    
    dir = (sin(incl) × cos(az), sin(incl) × sin(az), cos(incl))
    if dir · (0, 0, 1) ≥ cos_threshold:     // within forward FOV
        directions.push(dir)
```

`unobstructed_dir` iterates through these directions and returns the first one with no raycast hit within 4.0 units.

### Velocity & Position Integration

```
velocity += separation + alignment + cohesion + boundary_force [+ fleet_force]
velocity  = clamp(velocity, maxSpeed)
position += velocity × Δt
```

### Orientation

```
forward  = normalize(velocity)
rotation = rotation_arc(from = (0, 1, 0), to = forward)    // shortest rotation from up to forward
```

### Target Movement

```
target.position = (sin(t) × 5, cos(t) × 5, sin(t) × 3 × cos(t))
```

An orbiting Lissajous-like path.

### Initialization

```
B.position = (sin(i × 2) × 2, cos(i × 2) × 2, (i % 4) as f32)    // spiral ring
B.velocity = (0, 0, 0)
```

## 19

A Godot project skeleton for a planned Rust + BVH (bounding volume hierarchy) boids simulation. **No boids flocking math has been implemented** — only rendering infrastructure exists.

### Spawner Math

Boids are placed at random positions within a bounding box:

```
position = (rand(−bounds.x, bounds.x), rand(−bounds.y, bounds.y), rand(−bounds.z, bounds.z))
```

### Rendering Infrastructure

- A **GPU multi-mesh renderer** addon (`SimpleMultiMesh3D`/`SimpleMultiMesh2D`) batches transforms from grouped nodes into `MultiMesh` instances for efficient rendering
- No `.rs` Rust source files are present — the BVH-accelerated boids simulation was never coded

### Initialization

```
B.position = random in spawn bounding box
```

## 20

An Unreal Engine 5 GPU compute shader simulation using a **hashed grid** with **bitonic sort** for spatial partitioning (O(k) neighbor queries instead of O(n²)). Features **exponential heading smoothing**, **3D value noise** for speed variation, **linear ramp separation**, and a **home attraction** force pulling boids toward the origin.

### Constants

- `numBoids = 1000`
- `neighbourDistance = 10.0` — perception radius
- `separationDistance = 3.0` — separation zone radius
- `homeInnerRadius = 200.0` — radius within which home force is inactive
- `boidSpeed = 10.0` — base speed
- `boidSpeedVariation = 1.0` — noise speed multiplier range
- `boidRotationSpeed = 10.0` — heading smoothing rate
- `homeUrge = 0.1`, `separationUrge = 0.1`, `cohesionUrge = 0.01`, `alignmentUrge = 0.1`
- `gridCellSize = 5.0`, `cellSizeReciprocal = 0.2`
- `gridDimensions = (256, 256, 256)`
- `spawnRadius = 600.0`
- Thread group size: 256

### Euclidean Distance

```
d² = dot(a − b, a − b)
d = |a − b|
```

### Hashed Grid Spatial Partitioning

Three GPU buffers enable O(k) neighbor lookups (3×3×3 = 27 cells searched):

**Cell mapping:**
```
cellIndex = floor(position × cellSizeReciprocal)
flatIndex = cellIndex.x + cellIndex.y × gridDim.x + cellIndex.z × gridDim.x × gridDim.y
flatIndex = flatIndex mod cellOffsetBufferSize    // hash to buffer
```

**Data structures:**
1. `cellIndexBuffer[i]` — which (hashed) cell each boid belongs to
2. `particleIndexBuffer` — boid indices sorted by cell via **bitonic sort**
3. `cellOffsetBuffer[cell]` — first index in the sorted list for each cell (set via `InterlockedMin`)

**Neighbor iteration** for a boid in cell C:
```
for all 3×3×3 neighboring cells:
    iterator = cellOffsetBuffer[neighborCell]
    while iterator < numParticles:
        boidB = particleIndexBuffer[iterator]
        if cellIndexBuffer[boidB] != neighborCell: break    // end of this cell
        // ... process boidB ...
        iterator++
```

### Separation

Linear ramp repulsion — strongest at zero distance, zero at `separationDistance`:

```
if dist < separationDistance and dist > 0:
    d_gap = separationDistance − dist
    d̂ = (position_b − position_a) / dist
    separation −= d̂ × d_gap
```

Force magnitude is proportional to `separationDistance − dist`, i.e. how far inside the separation zone the neighbor is.

### Alignment

Sum of neighbor directions, normalized:

```
alignment = Σ directions[neighbor]    // accumulated in neighbor loop
alignment = safeNormal(alignment)     // normalize to unit vector
```

### Cohesion

Center of mass toward neighbors (including self as seed):

```
neighboursCentre = position_a    // seed with self
count = 1

for each neighbor within neighbourhoodDistance:
    neighboursCentre += position_b
    count++

cohesion = neighboursCentre / count − position_a    // vector toward COM
```

### Home Force

Attraction toward origin when outside the inner radius:

```
home = (0, 0, 0)
distFromHome = |home − position_a|

if distFromHome > homeInnerRadius:
    homeDir = normalize(home − position_a)
else:
    homeDir = (0, 0, 0)
```

### Direction Blending & Exponential Smoothing

All four forces are weighted and summed, then exponentially smoothed with the previous direction:

```
newDirection = alignment × alignmentUrge
             + separation × separationUrge
             + cohesion × cohesionUrge
             + homeDir × homeUrge

ip = exp(−boidRotationSpeed × dt)                     // interpolation parameter
newDirection = lerp(newDirection, currentDirection, ip)    // (1−ip)×new + ip×old
newDirection = safeNormal(newDirection)
```

When `dt → 0`, `ip → 1` → direction barely changes (smooth). When `dt` is large or `rotationSpeed` is high, `ip → 0` → direction changes quickly.

### 3D Value Noise (Speed Variation)

Per-boid time-varying speed using hashed 3D value noise:

```
hash(n) = frac(sin(n) × 43758.5453)

noise1(x):
    p = floor(x), f = frac(x)
    f = f² × (3 − 2f)                      // smoothstep interpolation
    n = p.x + p.y × 57 + 113 × p.z
    return trilinear_lerp(hash(n+000), hash(n+001), ..., hash(n+170), f)

noise = clamp(noise1(totalTime/100 + hash(index)), −1, 1) × 2 − 1
velocity = boidSpeed × (1 + noise × boidSpeedVariation)
```

Each boid gets a unique noise offset via `hash(index)`, creating varied individual speeds that change over time.

### Position Integration

```
position += direction × velocity × dt
```

### Look-At Matrix (Rendering)

Per-boid transform matrix from position and direction:

```
at   = position
eye  = position − normalize(direction)

zaxis = normalize(at − eye)          // forward
xaxis = normalize(cross(up, zaxis))
yaxis = cross(zaxis, xaxis)

mat = [xaxis yaxis zaxis translation] × particleScale
```

### Initialization

```
// Rejection sampling for uniform sphere:
repeat: s = (rand(−1,1), rand(−1,1), rand(−1,1))
until |s|² ≤ 1
position = s × spawnRadius

direction = randomUnitVector()
```

