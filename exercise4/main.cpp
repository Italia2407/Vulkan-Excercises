#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>
#include <chrono>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <stb_image_write.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "vertex_data.hpp"

namespace
{
using Clock_ = std::chrono::steady_clock;
using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define ASSETDIR_ "assets/exercise4/"
		constexpr char const* kFloorTexture = ASSETDIR_ "asphalt.png";
		constexpr char const* kSpriteTexture = ASSETDIR_ "explosion.png";

#		define SHADERDIR_ ASSETDIR_ "shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "shaderTex.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "shaderTex.frag.spv";

		constexpr char const* kAlphaFragShaderPath = SHADERDIR_ "shaderTexAlpha.frag.spv";
#		undef SHADERDIR_
#		undef ASSETDIR_



		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 5.0f;
		constexpr float kCameraSlowMult = 0.05f;

		constexpr float kCameraMouseSensitivity = 0.01f;
	}

	// GLFW callbacks
	void glfw_callback_key_press( GLFWwindow*, int, int, int, int );
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	enum class EInputState
	{
		forward,
		backward,
		leftward,
		rightward,
		upward,
		downward,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.0f;
		float mouseY = 0.0f;

		float previousX = 0.0f;
		float previousY = 0.0f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	void update_user_state(UserState&, float aElapsedTime);

	// Uniform data
	namespace glsl
	{
	struct SceneUniform
	{
		glm::mat4 camera;
		glm::mat4 projection;
		glm::mat4 projCam;
	};

	static_assert(sizeof(SceneUniform) <= 65536,
		"SceneUniform must be Less than 65536 Bytes for vkCmdUpdateBuffer()");
	static_assert(sizeof(SceneUniform) % 4 == 0,
		"SceneUniform Size must be a Multiple of 4 Bytes");
	}

	// Helpers:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& );

	lut::PipelineLayout create_pipeline_layout( lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout );
	lut::Pipeline create_pipeline( lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout );
	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( lut::VulkanWindow const&, lut::Allocator const& );

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const&
	);

	void record_commands( 
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkExtent2D const&,
		VkBuffer aPositionBuffer,
		VkBuffer aTextureCoordBuffer,
		std::uint32_t aVertexCount,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet aObjectDescriptors,
		VkBuffer aSpritePositionBuffer,
		VkBuffer aSpriteTextureBuffer,
		std::uint32_t aSpriteVertexCount,
		VkDescriptorSet aSpriteObjDescriptors,
		VkPipeline aAlphaPipeline
	);
	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	UserState userState{};
	glfwSetWindowUserPointer(window.window, &userState);

	// Configure the GLFW window
	glfwSetKeyCallback( window.window, &glfw_callback_key_press );
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);

	lut::PipelineLayout pipeLayout = create_pipeline_layout( window, sceneLayout.handle, objectLayout.handle );
	lut::Pipeline pipe = create_pipeline( window, renderPass.handle, pipeLayout.handle );
	lut::Pipeline alphaPipeline = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);

	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers, depthBufferView.handle );

	lut::CommandPool cpool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	// Load data
	TexturedMesh planeMesh = create_plane_mesh( window, allocator );
	TexturedMesh spriteMesh = create_sprite_mesh(window, allocator);

	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::DescriptorPool descriptorPool = lut::create_descriptor_pool(window);

	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, descriptorPool.handle, sceneLayout.handle);
	{
		VkDescriptorBufferInfo sceneUBOInfo{}; {
			sceneUBOInfo.buffer = sceneUBO.buffer;
			sceneUBOInfo.range = VK_WHOLE_SIZE;
		}

		VkWriteDescriptorSet writeDescriptorSets[1]{}; {
			writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

			writeDescriptorSets[0].dstSet = sceneDescriptors;
			writeDescriptorSets[0].dstBinding = 0;

			writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			writeDescriptorSets[0].descriptorCount = 1;
			writeDescriptorSets[0].pBufferInfo = &sceneUBOInfo;
		}

		constexpr auto numDescriptorSets = sizeof(writeDescriptorSets) / sizeof(writeDescriptorSets[0]);
		vkUpdateDescriptorSets(window.device, numDescriptorSets, writeDescriptorSets, 0, nullptr);
	}

	lut::Image floorTexture;
	lut::Image spriteTexture;
	{
		lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

		floorTexture = lut::load_image_texture2d(cfg::kFloorTexture, window, loadCmdPool.handle, allocator);
		spriteTexture = lut::load_image_texture2d(cfg::kSpriteTexture, window, loadCmdPool.handle, allocator);
	}
	lut::ImageView floorView = lut::create_image_view_texture2d(window, floorTexture.image, VK_FORMAT_R8G8B8A8_SRGB);
	lut::ImageView spriteView = lut::create_image_view_texture2d(window, spriteTexture.image, VK_FORMAT_R8G8B8A8_SRGB);

	lut::Sampler defaultSampler = lut::create_default_sampler(window);
	
	VkDescriptorSet floorDescriptors = lut::alloc_desc_set(window, descriptorPool.handle, objectLayout.handle);
	{
		VkDescriptorImageInfo textureInfo{}; {
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = floorView.handle;

			textureInfo.sampler = defaultSampler.handle;
		}

		VkWriteDescriptorSet descriptorSets[1]{}; {
			descriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

			descriptorSets[0].dstSet = floorDescriptors;
			descriptorSets[0].dstBinding = 0;

			descriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorSets[0].descriptorCount = 1;

			descriptorSets[0].pImageInfo = &textureInfo;
		}

		constexpr auto numSets = sizeof(descriptorSets) / sizeof(descriptorSets[0]);
		vkUpdateDescriptorSets(window.device, numSets, descriptorSets, 0, nullptr);
	}
	VkDescriptorSet spriteDescriptors = lut::alloc_desc_set(window, descriptorPool.handle, objectLayout.handle);
	{
		VkDescriptorImageInfo textureInfo{}; {
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = spriteView.handle;

			textureInfo.sampler = defaultSampler.handle;
		}

		VkWriteDescriptorSet descriptorSets[1]{}; {
			descriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

			descriptorSets[0].dstSet = spriteDescriptors;
			descriptorSets[0].dstBinding = 0;

			descriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorSets[0].descriptorCount = 1;

			descriptorSets[0].pImageInfo = &textureInfo;
		}

		constexpr auto numSets = sizeof(descriptorSets) / sizeof(descriptorSets[0]);
		vkUpdateDescriptorSets(window.device, numSets, descriptorSets, 0, nullptr);
	}
	
	// Application main loop
	bool recreateSwapchain = false;

	auto previousClock = Clock_::now();
	while( !glfwWindowShouldClose( window.window ) )
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents();

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			vkDeviceWaitIdle(window.device);

			auto const changes = lut::recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderPass = create_render_pass(window);
			}

			if (changes.changedSize)
			{
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				alphaPipeline = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);
			
			recreateSwapchain = false;
			continue;
		}

		std::uint32_t imageIndex = 0;
		auto const aquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (aquireRes == VK_SUBOPTIMAL_KHR || aquireRes == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain = true;
			continue;
		}
		else if (aquireRes != VK_SUCCESS)
		{
			throw lut::Error("Unable to Acquire enxt Swapchain Image\n"
				"vkAcquireNextImageKHR() Returned %s", lut::to_string(aquireRes).c_str());
		}

		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Wait for Command Buffer Fence %u\n"
				"vkWaitForFences() Returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Reset Command Buffer Fence %u\n"
				"vkResetFences() Returned %s", imageIndex, lut::to_string(res).c_str());
		}
		
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		auto const now = Clock_::now();
		auto const deltaTime = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(userState, deltaTime);

		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, userState);

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			window.swapchainExtent,
			planeMesh.positions.buffer, planeMesh.textureCoords.buffer,
			planeMesh.vertexCount,
			sceneUBO.buffer,
			sceneUniforms,
			pipeLayout.handle,
			sceneDescriptors,
			floorDescriptors,
			spriteMesh.positions.buffer,
			spriteMesh.textureCoords.buffer,
			spriteMesh.vertexCount,
			spriteDescriptors,
			alphaPipeline.handle
		);
		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		VkPresentInfoKHR presentInfo{}; {
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = &renderFinished.handle;

			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = &window.swapchain;

			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr;
		}

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);
		if (presentRes == VK_SUBOPTIMAL_KHR || presentRes == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain = true;
		}
		else if (presentRes != VK_SUCCESS)
		{
			throw lut::Error("Unable to Present Swapchain Image %u\n"
				"vkQueuePresentKHR() Returned %s", imageIndex, lut::to_string(presentRes).c_str());
		}
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
{
	if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
	{
		glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
	}

	auto userState = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
	assert(userState);

	bool const isReleased = (GLFW_RELEASE == aAction);

	switch (aKey)
	{
	case GLFW_KEY_W:
		userState->inputMap[std::size_t(EInputState::forward)] = !isReleased;
		break;
	case GLFW_KEY_S:
		userState->inputMap[std::size_t(EInputState::backward)] = !isReleased;
		break;
	case GLFW_KEY_A:
		userState->inputMap[std::size_t(EInputState::leftward)] = !isReleased;
		break;
	case GLFW_KEY_D:
		userState->inputMap[std::size_t(EInputState::rightward)] = !isReleased;
		break;
	case GLFW_KEY_E:
		userState->inputMap[std::size_t(EInputState::upward)] = !isReleased;
		break;
	case GLFW_KEY_Q:
		userState->inputMap[std::size_t(EInputState::downward)] = !isReleased;
		break;
	
	case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
	case GLFW_KEY_RIGHT_SHIFT:
		userState->inputMap[std::size_t(EInputState::fast)] = !isReleased;
		break;

	case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
	case GLFW_KEY_RIGHT_CONTROL:
		userState->inputMap[std::size_t(EInputState::slow)] = !isReleased;
		break;
	}
}
void glfw_callback_button(GLFWwindow* aWindow, int aButton, int aAction, int)
{
	auto userState = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
	assert(userState);

	if (GLFW_MOUSE_BUTTON_RIGHT == aButton && GLFW_PRESS == aAction)
	{
		auto& flag = userState->inputMap[std::size_t(EInputState::mousing)];

		flag = !flag;
		if (flag)
			glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		else
			glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}
void glfw_callback_motion(GLFWwindow* aWindow, double aXPos, double aYPos)
{
	auto userState = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
	assert(userState);

	userState->mouseX = aXPos;
	userState->mouseY = aYPos;
}
}

namespace
{
void update_user_state(UserState& aUserState, float aElapsedTime)
{
	auto& camera = aUserState.camera2world;

	if (aUserState.inputMap[std::size_t(EInputState::mousing)])
	{
		if (aUserState.wasMousing)
		{
			auto const sensitivity = cfg::kCameraMouseSensitivity;
			auto const dX = sensitivity * (aUserState.mouseX - aUserState.previousX);
			auto const dY = sensitivity * (aUserState.mouseY - aUserState.previousY);

			camera = camera * glm::rotate(-dY, glm::vec3(1.0f, 0.0f, 0.0f));
			camera = camera * glm::rotate(-dX, glm::vec3(0.0f, 1.0f, 0.0f));
		}

		aUserState.previousX = aUserState.mouseX;
		aUserState.previousY = aUserState.mouseY;
		aUserState.wasMousing = true;
	}
	else
	{
		aUserState.wasMousing = false;
	}

	auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
		(aUserState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.0f) *
		(aUserState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.0f);

	if (aUserState.inputMap[std::size_t(EInputState::forward)])
		camera = camera * glm::translate(glm::vec3(0.0f, 0.0f, -move));
	if (aUserState.inputMap[std::size_t(EInputState::backward)])
		camera = camera * glm::translate(glm::vec3(0.0f, 0.0f,  move));

	if (aUserState.inputMap[std::size_t(EInputState::leftward)])
		camera = camera * glm::translate(glm::vec3(-move, 0.0f, 0.0f));
	if (aUserState.inputMap[std::size_t(EInputState::rightward)])
		camera = camera * glm::translate(glm::vec3( move, 0.0f, 0.0f));

	if (aUserState.inputMap[std::size_t(EInputState::upward)])
		camera = camera * glm::translate(glm::vec3(0.0f,  move, 0.0f));
	if (aUserState.inputMap[std::size_t(EInputState::downward)])
		camera = camera * glm::translate(glm::vec3(0.0f, -move, 0.0f));
}

void update_scene_uniforms( glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aUserState )
{
	float const aspect = aFramebufferWidth / float(aFramebufferHeight);

	aSceneUniforms.projection = glm::perspectiveRH_ZO(
		lut::Radians(cfg::kCameraFov).value(),
		aspect,
		cfg::kCameraNear, cfg::kCameraFar
	);
	aSceneUniforms.projection[1][1] *= -1.0f;

	aSceneUniforms.camera = glm::translate(glm::vec3(0.0f, -0.3f, -1.0f));
	aSceneUniforms.camera = glm::inverse(aUserState.camera2world);

	aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
}
}

namespace
{
lut::RenderPass create_render_pass( lut::VulkanWindow const& aWindow )
{
	VkAttachmentDescription attachments[2]{}; {
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	}

	VkAttachmentReference subpassAttachments[1]{}; {
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}
	VkAttachmentReference depthAttachment{}; {
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	}
	
	VkSubpassDescription subpasses[1]{}; {
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;

		subpasses[0].pDepthStencilAttachment = &depthAttachment;
	}

	VkRenderPassCreateInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

		renderPassInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
		renderPassInfo.pAttachments = attachments;

		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = subpasses;

		renderPassInfo.dependencyCount = 0;
		renderPassInfo.pDependencies = nullptr;
	}

	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (auto const res = vkCreateRenderPass(aWindow.device, &renderPassInfo, nullptr, &renderPass); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Render Pass\n"
			"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
	}

	return lut::RenderPass(aWindow.device, renderPass);
}

lut::PipelineLayout create_pipeline_layout( lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout )
{
	VkDescriptorSetLayout descriptorSetLayouts[] = {
		aSceneLayout,
		aObjectLayout
	};

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{}; {
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		pipelineLayoutInfo.setLayoutCount = sizeof(descriptorSetLayouts) / sizeof(descriptorSetLayouts[0]);
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts;

		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;
	}

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	if (auto const res = vkCreatePipelineLayout(aContext.device, &pipelineLayoutInfo, nullptr, &pipelineLayout); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Pipeline Layout\n"
			"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
	}

	return lut::PipelineLayout(aContext.device, pipelineLayout);
}
lut::Pipeline create_pipeline( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout )
{
	lut::ShaderModule vertShader = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
	lut::ShaderModule fragShader = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

	VkPipelineShaderStageCreateInfo shaderStagesInfo[2]{}; {
		shaderStagesInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[0].pName = "main";

		shaderStagesInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderStagesInfo[0].module = vertShader.handle;

		shaderStagesInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[1].pName = "main";

		shaderStagesInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderStagesInfo[1].module = fragShader.handle;
	}

	VkVertexInputBindingDescription vertexInputBindings[2]{}; {
		vertexInputBindings[0].binding = 0;
		vertexInputBindings[0].stride = sizeof(float) * 3;
		vertexInputBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	
		vertexInputBindings[1].binding = 1;
		vertexInputBindings[1].stride = sizeof(float) * 2;
		vertexInputBindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}

	VkVertexInputAttributeDescription vertexInputAttributes[2]{}; {
		vertexInputAttributes[0].binding = 0;
		vertexInputAttributes[0].location = 0;
		vertexInputAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributes[0].offset = 0;

		vertexInputAttributes[1].binding = 1;
		vertexInputAttributes[1].location = 1;
		vertexInputAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexInputAttributes[1].offset = 0;
	}

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; {
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	
		vertexInputInfo.vertexBindingDescriptionCount = 2;
		vertexInputInfo.pVertexBindingDescriptions = vertexInputBindings;

		vertexInputInfo.vertexAttributeDescriptionCount = 2;
		vertexInputInfo.pVertexAttributeDescriptions = vertexInputAttributes;
	}
	VkPipelineInputAssemblyStateCreateInfo assemblyStateInfo{}; {
		assemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

		assemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyStateInfo.primitiveRestartEnable = VK_FALSE;
	}
	
	VkViewport viewport{}; {
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);

		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	VkRect2D scissor{}; {
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;
	}
	VkPipelineViewportStateCreateInfo viewportStateInfo{}; {
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pViewports = &viewport;

		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.pScissors = &scissor;
	}
	
	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{}; {
		rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		rasterizationStateInfo.depthClampEnable = VK_FALSE;
		rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateInfo.depthBiasEnable = VK_FALSE;

		rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		rasterizationStateInfo.lineWidth = 1.0f;
	}
	
	VkPipelineMultisampleStateCreateInfo multisamplingStateInfo{}; {
		multisamplingStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

		multisamplingStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	}
	
	VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{}; {
		depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

		depthStencilStateInfo.depthTestEnable = VK_TRUE;
		depthStencilStateInfo.depthWriteEnable = VK_TRUE;
		depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

		depthStencilStateInfo.minDepthBounds = 0.0f;
		depthStencilStateInfo.maxDepthBounds = 1.0f;
	}

	VkPipelineColorBlendAttachmentState colourBlendAttachmentStates[1]{}; {
		colourBlendAttachmentStates[0].blendEnable = VK_FALSE;
		colourBlendAttachmentStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}
	VkPipelineColorBlendStateCreateInfo colourBlendStateInfo{}; {
		colourBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		colourBlendStateInfo.logicOpEnable = VK_FALSE;

		colourBlendStateInfo.attachmentCount = 1;
		colourBlendStateInfo.pAttachments = colourBlendAttachmentStates;
	}
	
	VkGraphicsPipelineCreateInfo graphicsPipelineInfo{}; {
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStagesInfo;

		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &assemblyStateInfo;
		graphicsPipelineInfo.pTessellationState = nullptr;
		graphicsPipelineInfo.pViewportState = &viewportStateInfo;
		graphicsPipelineInfo.pRasterizationState = &rasterizationStateInfo;
		graphicsPipelineInfo.pMultisampleState = &multisamplingStateInfo;
		graphicsPipelineInfo.pDepthStencilState = &depthStencilStateInfo;
		graphicsPipelineInfo.pColorBlendState = &colourBlendStateInfo;
		graphicsPipelineInfo.pDynamicState = nullptr;

		graphicsPipelineInfo.layout = aPipelineLayout;
		graphicsPipelineInfo.renderPass = aRenderPass;
		graphicsPipelineInfo.subpass = 0;
	}
	
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;
	if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &graphicsPipeline); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Graphics Pipeline\n"
			"vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
	}
	
	return lut::Pipeline(aWindow.device, graphicsPipeline);
}
lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
{
	lut::ShaderModule vertShader = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
	lut::ShaderModule fragShader = lut::load_shader_module(aWindow, cfg::kAlphaFragShaderPath);

	VkPipelineShaderStageCreateInfo shaderStagesInfo[2]{}; {
		shaderStagesInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[0].pName = "main";

		shaderStagesInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderStagesInfo[0].module = vertShader.handle;

		shaderStagesInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[1].pName = "main";

		shaderStagesInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderStagesInfo[1].module = fragShader.handle;
	}

	VkVertexInputBindingDescription vertexInputBindings[2]{}; {
		vertexInputBindings[0].binding = 0;
		vertexInputBindings[0].stride = sizeof(float) * 3;
		vertexInputBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	
		vertexInputBindings[1].binding = 1;
		vertexInputBindings[1].stride = sizeof(float) * 2;
		vertexInputBindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}

	VkVertexInputAttributeDescription vertexInputAttributes[2]{}; {
		vertexInputAttributes[0].binding = 0;
		vertexInputAttributes[0].location = 0;
		vertexInputAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributes[0].offset = 0;

		vertexInputAttributes[1].binding = 1;
		vertexInputAttributes[1].location = 1;
		vertexInputAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexInputAttributes[1].offset = 0;
	}

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; {
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	
		vertexInputInfo.vertexBindingDescriptionCount = 2;
		vertexInputInfo.pVertexBindingDescriptions = vertexInputBindings;

		vertexInputInfo.vertexAttributeDescriptionCount = 2;
		vertexInputInfo.pVertexAttributeDescriptions = vertexInputAttributes;
	}
	VkPipelineInputAssemblyStateCreateInfo assemblyStateInfo{}; {
		assemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

		assemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyStateInfo.primitiveRestartEnable = VK_FALSE;
	}
	
	VkViewport viewport{}; {
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);

		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	VkRect2D scissor{}; {
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;
	}
	VkPipelineViewportStateCreateInfo viewportStateInfo{}; {
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pViewports = &viewport;

		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.pScissors = &scissor;
	}
	
	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{}; {
		rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		rasterizationStateInfo.depthClampEnable = VK_FALSE;
		rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateInfo.depthBiasEnable = VK_FALSE;

		rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		rasterizationStateInfo.lineWidth = 1.0f;
	}
	
	VkPipelineMultisampleStateCreateInfo multisamplingStateInfo{}; {
		multisamplingStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

		multisamplingStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	}
	
	VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{}; {
		depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

		depthStencilStateInfo.depthTestEnable = VK_TRUE;
		depthStencilStateInfo.depthWriteEnable = VK_TRUE;
		depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

		depthStencilStateInfo.minDepthBounds = 0.0f;
		depthStencilStateInfo.maxDepthBounds = 1.0f;
	}

	VkPipelineColorBlendAttachmentState colourBlendAttachmentStates[1]{}; {
		colourBlendAttachmentStates[0].blendEnable = VK_TRUE;

		colourBlendAttachmentStates[0].colorBlendOp = VK_BLEND_OP_ADD;

		colourBlendAttachmentStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colourBlendAttachmentStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		colourBlendAttachmentStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}
	VkPipelineColorBlendStateCreateInfo colourBlendStateInfo{}; {
		colourBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		colourBlendStateInfo.logicOpEnable = VK_FALSE;

		colourBlendStateInfo.attachmentCount = 1;
		colourBlendStateInfo.pAttachments = colourBlendAttachmentStates;
	}
	
	VkGraphicsPipelineCreateInfo graphicsPipelineInfo{}; {
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStagesInfo;

		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &assemblyStateInfo;
		graphicsPipelineInfo.pTessellationState = nullptr;
		graphicsPipelineInfo.pViewportState = &viewportStateInfo;
		graphicsPipelineInfo.pRasterizationState = &rasterizationStateInfo;
		graphicsPipelineInfo.pMultisampleState = &multisamplingStateInfo;
		graphicsPipelineInfo.pDepthStencilState = &depthStencilStateInfo;
		graphicsPipelineInfo.pColorBlendState = &colourBlendStateInfo;
		graphicsPipelineInfo.pDynamicState = nullptr;

		graphicsPipelineInfo.layout = aPipelineLayout;
		graphicsPipelineInfo.renderPass = aRenderPass;
		graphicsPipelineInfo.subpass = 0;
	}
	
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;
	if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &graphicsPipeline); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Graphics Pipeline\n"
			"vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
	}
	
	return lut::Pipeline(aWindow.device, graphicsPipeline);
}

std::tuple<lut::Image, lut::ImageView> create_depth_buffer( lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator )
{
	VkImageCreateInfo imageInfo{}; {
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;

		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;

		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;

		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	VmaAllocationCreateInfo allocationInfo{}; {
		allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	}

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocationInfo, &image, &allocation, nullptr);
		res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Allocate Depth Buffer Image\n"
			"vmaCreateImage() Returned %s", lut::to_string(res).c_str());
	}

	lut::Image depthImage = lut::Image(aAllocator.allocator, image, allocation);

	VkImageViewCreateInfo imageViewInfo{}; {
		imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

		imageViewInfo.image = depthImage.image;
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

		imageViewInfo.format = cfg::kDepthFormat;
		imageViewInfo.components = VkComponentMapping{};
		imageViewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};
	}

	VkImageView imageView = VK_NULL_HANDLE;
	if (auto const res = vkCreateImageView(aWindow.device, &imageViewInfo, nullptr, &imageView);
		res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Image View\n"
			"vkCreateImageView() Returned %s", lut::to_string(res).c_str());
	}

	return {std::move(depthImage), lut::ImageView(aWindow.device, imageView)};
}

void create_swapchain_framebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView )
{
	assert( aFramebuffers.empty() );

	for (std::size_t i = 0; i < aWindow.swapViews.size(); i++)
	{
		VkImageView attachments[2] = {
			aWindow.swapViews[i],
			aDepthView
		};

		VkFramebufferCreateInfo framebufferInfo{}; {
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

			framebufferInfo.flags = 0;
			framebufferInfo.renderPass = aRenderPass;

			framebufferInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
			framebufferInfo.pAttachments = attachments;

			framebufferInfo.width = aWindow.swapchainExtent.width;
			framebufferInfo.height = aWindow.swapchainExtent.height;

			framebufferInfo.layers = 1;
		}

		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &framebufferInfo, nullptr, &framebuffer); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Create Framebuffer for Swapchain Image %zu\n"
				"vkCreateFramebuffer() Returned %s", i, lut::to_string(res).c_str());
		}
		aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, framebuffer));
	}

	assert( aWindow.swapViews.size() == aFramebuffers.size() );
}

lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& aWindow )
{
	VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[1]{}; {
		descriptorSetLayoutBindings[0].binding = 0;

		descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorSetLayoutBindings[0].descriptorCount = 1;
		descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	}
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{}; {
		descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		descriptorSetLayoutInfo.bindingCount = sizeof(descriptorSetLayoutBindings) / sizeof(descriptorSetLayoutBindings[0]);
		descriptorSetLayoutInfo.pBindings = descriptorSetLayoutBindings;
	}

	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout);
		res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Descriptor Set Layout\n"
			"vkCreateDescriptorSetLayout() Returned %s", lut::to_string(res).c_str());
	}

	return lut::DescriptorSetLayout(aWindow.device, descriptorSetLayout);
}
lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& aWindow )
{
	VkDescriptorSetLayoutBinding layoutBindings[1]{}; {
		layoutBindings[0].binding = 0;

		layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		layoutBindings[0].descriptorCount = 1;

		layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	VkDescriptorSetLayoutCreateInfo layoutInfo{}; {
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		layoutInfo.bindingCount = sizeof(layoutBindings) / sizeof(layoutBindings[0]);
		layoutInfo.pBindings = layoutBindings;
	}

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
		res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Descriptor Set Layout\n"
			"vkCreateDescriptorSetLayout() Returned %s", lut::to_string(res).c_str());
	}

	return lut::DescriptorSetLayout(aWindow.device, layout);
}

void record_commands(
	VkCommandBuffer aCmdBuff,
	VkRenderPass aRenderPass,
	VkFramebuffer aFramebuffer,
	VkPipeline aGraphicsPipe,
	VkExtent2D const& aImageExtent,
	VkBuffer aPositionBuffer, VkBuffer aTextureCoordBuffer,
	std::uint32_t aVertexCount,
	VkBuffer aSceneUBO,
	glsl::SceneUniform const& aSceneUniform,
	VkPipelineLayout aGraphicsLayout,
	VkDescriptorSet aSceneDescriptors,
	VkDescriptorSet aObjectDescriptors,
	VkBuffer aSpritePositionBuffer,
	VkBuffer aSpriteTextureBuffer,
	std::uint32_t aSpriteVertexCount,
	VkDescriptorSet aSpriteObjDescriptors,
	VkPipeline aAlphaPipeline)
{
	// Begin Recording Commands
	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(aCmdBuff, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Begin Recording Command Buffer\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	lut::buffer_barrier(
		aCmdBuff,
		aSceneUBO,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT
	);

	vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

	lut::buffer_barrier(
		aCmdBuff,
		aSceneUBO,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
	);

	// Begin the Render Pass
	VkClearValue clearValues[2]{}; {
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;

		clearValues[1].depthStencil.depth = 1.0f;
	}
	
	VkRenderPassBeginInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

		renderPassInfo.renderPass = aRenderPass;
		renderPassInfo.framebuffer = aFramebuffer;

		renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
		renderPassInfo.renderArea.extent = aImageExtent;

		renderPassInfo.clearValueCount = sizeof(clearValues) / sizeof(clearValues[0]);
		renderPassInfo.pClearValues = clearValues;
	}

	vkCmdBeginRenderPass(aCmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Begin Drawing with our Graphics Pipeline
	vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);

	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aObjectDescriptors, 0, nullptr);

	VkBuffer buffers[2] = {aPositionBuffer, aTextureCoordBuffer};
	VkDeviceSize offsets[2]{};

	vkCmdBindVertexBuffers(aCmdBuff, 0, 2, buffers, offsets);
	vkCmdDraw(aCmdBuff, aVertexCount, 1, 0, 0);

	vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aAlphaPipeline);

	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aSpriteObjDescriptors, 0, nullptr);

	VkBuffer spriteBuffers[2] = {aSpritePositionBuffer, aSpriteTextureBuffer};
	VkDeviceSize spriteOffsets[2]{};

	vkCmdBindVertexBuffers(aCmdBuff, 0, 2, spriteBuffers, spriteOffsets);
	vkCmdDraw(aCmdBuff, aSpriteVertexCount, 1, 0, 0);

	// End the Render Pass
	vkCmdEndRenderPass(aCmdBuff);

	// End Command Recording
	if (auto const res = vkEndCommandBuffer(aCmdBuff); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to End Recording Command Buffer\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}
}
void submit_commands( lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
{
	VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;
	}

	if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Submit Command Buffer to Queue\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}
}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		throw lut::Error( "Not yet implemented" ); //TODO: (Section 1/Exercise 3) implement me!
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
