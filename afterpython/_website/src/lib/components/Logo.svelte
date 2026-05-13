<script lang="ts">
	import { asset, resolve } from "$app/paths";

	type LogoProps = {
		class?: string;
		size?: 'sm' | 'md' | 'lg' | 'xl';
		text?: string;
		showText?: boolean;
	}

	let { class: className = '', size = 'md', text = '', showText = true }: LogoProps = $props();

	let hasLogo = $state(true);

	// Size classes for logo image
	const sizeClasses = {
		sm: 'h-10',
		md: 'h-14',
		lg: 'h-18',
		xl: 'h-22'
	};

	// Font size classes for logo text - matching MyST's responsive sizing
	const textSizeClasses = {
		sm: 'text-md sm:text-lg tracking-tight',
		md: 'text-md sm:text-xl tracking-tight',
		lg: 'text-lg sm:text-2xl tracking-tight',
		xl: 'text-xl sm:text-3xl tracking-tight'
	};

	// Handle fallback to different image formats
	function handleImageError(event: Event) {
		const img = event.target as HTMLImageElement;
		const currentSrc = img.src;

		// Try formats in priority order: svg -> png -> jpg -> jpeg
		if (currentSrc.endsWith('/logo.svg')) {
			img.src = asset('/logo.png');
		} else if (currentSrc.endsWith('/logo.png')) {
			img.src = asset('/logo.jpg');
		} else if (currentSrc.endsWith('/logo.jpg')) {
			img.src = asset('/logo.jpeg');
		} else {
			// All formats failed, hide the logo
			hasLogo = false;
		}
	}
</script>

{#if hasLogo}
	<a href={resolve('/')} class="flex items-center gap-3 {className}">
		<img
			src={asset('/logo.svg')}
			alt="Logo"
			class="{sizeClasses[size]} w-auto"
			onerror={handleImageError}
		/>
		{#if showText && text}
			<span class="{textSizeClasses[size]}">
				{text}
			</span>
		{/if}
	</a>
{/if}
