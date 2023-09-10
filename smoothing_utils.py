import numpy as np
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# 1D
def circulantshift(xs, h):
	return np.hstack([xs[h:], xs[:h]] if h > 0 else [xs[h:], xs[:h]])

def circulant_dx(xs, h):
	return (circulantshift(xs, h) - xs)

def psf2otf(psf, N):
	pad = np.zeros((N,))
	n = len(psf)
	pad[:n] = psf
	pad = np.concatenate([pad[n/2:], pad[:n/2]])
	otf = fft(pad)
	return otf

# 2D
def circulantshift2_x(xs, h):
	return np.hstack([xs[:, h:], xs[:, :h]] if h > 0 else [xs[:, h:], xs[:, :h]])

def circulantshift2_y(xs, h):
	return np.vstack([xs[h:, :], xs[:h, :]] if h > 0 else [xs[h:, :], xs[:h, :]])

def circulant2_dx(xs, h):
	return (circulantshift2_x(xs, h) - xs)

def circulant2_dy(xs, h):
	return (circulantshift2_y(xs, h) - xs)

def l0_gradient_minimization_1d(I, lmd, beta_max, beta_rate=2.0, max_iter=30, return_history=False):
	##configuration
	lmd = 0.02
	
	S = np.array(I).ravel()
	
	# prepare FFT
	F_I = fft(S)
	F_denom = np.abs(psf2otf([-1, 1], S.shape[0]))**2.0

	# optimization
	S_history = [S]
	beta = lmd*2.0
	hp = np.zeros_like(S)
	for i in range(max_iter):
		# with S, solve for hp in Eq. (12)
		hp = circulant_dx(S, 1)
		mask = hp**2.0 < lmd/beta
		hp[mask] = 0.0

		# with hp, solve for S in Eq. (8)
		S = np.real(ifft((F_I + beta*fft(circulant_dx(hp, -1))) / (1.0 + beta*F_denom)))

		# iteration step
		if return_history:
			S_history.append(np.array(S))
		beta *= beta_rate
		if beta > beta_max: break

	if return_history:
		return S_history

	return S

def l0_gradient_minimization_2d(I, lmd=0.02, beta_max=1.0e5, beta_rate=2.0, max_iter=30, return_history=False):
	u'''image I can be both 1ch (ndim=2) or D-ch (ndim=D)'''
	S = np.array(I)
	
	# prepare FFT
	F_I = fft2(S, axes=(0, 1))
	Ny, Nx = S.shape[:2]
	D = S.shape[2] if S.ndim == 3 else 1
	dx, dy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
	dx[int(Ny/2), int(Nx/2)-1:int(Nx/2)+1] = [-1, 1]
	dy[int(Ny/2)-1:int(Ny/2)+1, int(Nx/2)] = [-1, 1]
	F_denom = np.abs(fft2(dx))**2.0 + np.abs(fft2(dy))**2.0
	if D > 1: F_denom = np.dstack([F_denom]*D)

	S_history = [S]
	beta = lmd * 2.0
	hp, vp = np.zeros_like(S), np.zeros_like(S)
	for i in range(max_iter):
		# with S, solve for hp and vp in Eq. (12)
		hp, vp = circulant2_dx(S, 1), circulant2_dy(S, 1)
		if D == 1:
			mask = hp**2.0 + vp**2.0 < lmd/beta
		else:
			mask = np.sum(hp**2.0 + vp**2.0, axis=2) < lmd/beta
		hp[mask] = 0.0
		vp[mask] = 0.0

		# with hp and vp, solve for S in Eq. (8)
		hv = circulant2_dx(hp, -1) + circulant2_dy(vp, -1)
		S = np.real(ifft2((F_I + (beta*fft2(hv, axes=(0, 1))))/(1.0 + beta*F_denom), axes=(0, 1)))

		# iteration step
		if return_history:
			S_history.append(np.array(S))
		beta *= beta_rate
		if beta > beta_max: break

	if return_history:
		return S_history

	return S

def l0_gradient_minimization_test():
	# 1D test
	xs = np.linspace(-2, 2, 200)
	us = (np.arange(len(xs)) / 23) % 3

	def create_noisy_signal(us, noise):
		return us + np.random.randn(len(us)) * noise
	us_noisy = create_noisy_signal(us, 0.1)

	lmd = 0.015
	beta_max = 1.0e5
	beta_rate = 2.0
	us_denoise = l0_gradient_minimization_1d(us_noisy, lmd, beta_max, beta_rate)
	us_history = l0_gradient_minimization_1d(us_noisy, lmd, beta_max, beta_rate, 30, True)

	fig, axs = plt.subplots(5, 1)
	fig.suptitle((r'$L_0$ Gradient Minimization on 1D Array.' + '\n'
		+ r'$\lambda={:.3}, \beta_{{max}}={:.2e}, \kappa={:.3f}$').format(
		lmd, beta_max, beta_rate),
		fontsize=16)
	axs[0].plot(xs, us, color='red', label='org', linestyle='--', linewidth=4, alpha=0.5)
	axs[0].plot(xs, us_noisy, color='blue', label='noisy', linewidth=2, alpha=0.5)
	axs[0].plot(xs, us_denoise, color='black', label='denoise', linewidth=2, alpha=0.8)
	axs[0].legend()

	for i, ax in enumerate(axs[1:]):
		it = 5*i + 1
		if it < len(us_history):
			us_denoise = us_history[it]
			ax.plot(xs, us, color='red', label='org', linestyle='--', linewidth=1, alpha=0.5)
			ax.plot(xs, us_denoise, label='iter %d' % it, linewidth=1, alpha=0.5)
			ax.legend()

	# 2D test
	img, (lmd, beta_max, beta_rate), _ = get_configuration()
	sigma = 0.06
	#img_noise = add_noise(img, sigma)

	fig, axs = plt.subplots(2, 3, figsize=(12, 8))
	fig.suptitle((r'$L_0$ Gradient Minimization on 2D Image (noise $\sigma={:.3}$).' + '\n'
		+ r'$\beta_{{max}}={:.2e}, \kappa={:.3f}$').format(sigma, beta_max, beta_rate),
		fontsize=16)
	axs[0, 0].imshow(img)
	axs[0, 0].set_title('original')
	axs[0, 1].imshow(img_noise)
	axs[0, 1].set_title('noise')

	denoise_axs = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
	for ax, lmd in zip(denoise_axs, [0.002, 0.005, 0.02, 0.05]):
		#res = l0_gradient_minimization_2d(img_noise, lmd, beta_max, beta_rate)
		res = l0_gradient_minimization_2d(img, lmd, beta_max, beta_rate)
		ax.imshow(np.clip(res, 0, 1), interpolation='nearest')
		#ax.imshow(clip_img(res), interpolation='nearest')
		ax.set_title('denoise $\\lambda = %f$' % lmd)