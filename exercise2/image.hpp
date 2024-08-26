#pragma once

#include <volk/volk.h>

class Image final
{
	public:
		Image() noexcept, ~Image();

		explicit Image( VkDevice, VkImage = VK_NULL_HANDLE, VkDeviceMemory = VK_NULL_HANDLE ) noexcept;

		Image( Image const& ) = delete;
		Image& operator= (Image const&) = delete;

		Image( Image&& ) noexcept;
		Image& operator = (Image&&) noexcept;

	public:
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;

	private:
		VkDevice mDevice = VK_NULL_HANDLE;
};

