## Description

Study of Cosmic Ray Ionization rates along particle motion

## CR Populations

Calculating different Populations

### Backward moving particles 

($-1 < \mu < \mu_h$) where $\mu_h$ is at the highest peak

### Forward moving particles 

($-1 < \mu < \mu_l$) where $\mu_l$ is at the lowest peak 

such that $s_h$ and $s_l$ form a pocket

### Mirrored particles

($\mu_l < \mu < \mu_h$) where $\mu_l$ is at the lowest peak 

such that $s_h$ and $s_l$ form a pocket

## Approach

So the ionization in the pockets will have to be measured taking into account the number of pockets present in the trajectory.

For a certain number of pockets we will have these three populations having an impact on the ionization within the region. So we have to figure out a way to be able to:

- Identify the pockets on a specific trajectory
- Calculate the $\mu_l,\mu_h$ that bound the pocket 

Make sure there is a clause that stops integration on $s$ for $sin \alpha_i \leq \sqrt{B(s)/B(s')}$ during the column density integration, this along with a flag to trunc the ionization loop and continue with following value of $\alpha_i$ for a $s'$ which is the point of the particle being mirrored for that pitch angle