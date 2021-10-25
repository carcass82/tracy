/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "material.h"
#include "GL/glew.h"

class GLMaterial
{
public:
	
    GLMaterial()
    {
    }

	GLMaterial(const Material& material)
        : albedo_{ material.GetAlbedo() }
        , roughness_{ material.GetRoughness() }
        , metalness_{ material.GetMetalness() }
        , ior_{ material.GetRefractiveIndex() }
        , emissive_{ material.GetEmissive() }
        , translucent_{ material.GetTranslucent() }
        , base_color_map_{ material.GetTexture(Material::TextureID::eBASECOLOR) }
        , normal_map_{ material.GetTexture(Material::TextureID::eNORMAL) }
        , roughness_map_{ material.GetTexture(Material::TextureID::eROUGHNESS) }
        , metalness_map_{ material.GetTexture(Material::TextureID::eMETALNESS) }
        , emissive_map_{ material.GetTexture(Material::TextureID::eEMISSIVE) }
	{
	}

	void Draw(GLuint program, const vector<GLuint>& textures) const
	{
		glUniform3fv(glGetUniformLocation(program, "material.albedo"), 1, value_ptr(albedo_));
        glUniform3fv(glGetUniformLocation(program, "material.emissive"), 1, value_ptr(emissive_));
        glUniform1f(glGetUniformLocation(program, "material.roughness"), roughness_);
        glUniform1f(glGetUniformLocation(program, "material.metalness"), metalness_);
        glUniform1f(glGetUniformLocation(program, "material.ior"), ior_);
        glUniform1f(glGetUniformLocation(program, "material.translucent"), translucent_);

        glUniform1f(glGetUniformLocation(program, "textures.hasBaseColor"), 0);
        if (base_color_map_ != UINT32_MAX)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textures[base_color_map_]);

            glUniform1f(glGetUniformLocation(program, "textures.hasBaseColor"), 1);
            glUniform1i(glGetUniformLocation(program, "textures.baseColor"), 0);
        }

        glUniform1f(glGetUniformLocation(program, "textures.hasNormal"), 0);
        if (normal_map_ != UINT32_MAX)
        {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textures[normal_map_]);

            glUniform1f(glGetUniformLocation(program, "textures.hasNormal"), 1);
            glUniform1i(glGetUniformLocation(program, "textures.normal"), 1);
        }

        glUniform1f(glGetUniformLocation(program, "textures.hasRoughness"), 0);
        if (roughness_map_ != UINT32_MAX)
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, textures[roughness_map_]);

            glUniform1f(glGetUniformLocation(program, "textures.hasRoughness"), 1);
            glUniform1i(glGetUniformLocation(program, "textures.roughness"), 2);
        }

        glUniform1f(glGetUniformLocation(program, "textures.hasMetalness"), 0);
        if (metalness_map_ != UINT32_MAX)
        {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, textures[metalness_map_]);

            glUniform1f(glGetUniformLocation(program, "textures.hasMetalness"), 1);
            glUniform1i(glGetUniformLocation(program, "textures.metalness"), 3);
        }

        glUniform1f(glGetUniformLocation(program, "textures.hasEmissive"), 0);
        if (emissive_map_ != UINT32_MAX)
        {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, textures[emissive_map_]);

            glUniform1f(glGetUniformLocation(program, "textures.hasEmissive"), 1);
            glUniform1i(glGetUniformLocation(program, "textures.emissive"), 4);
        }
	}


private:

    vec3 albedo_{};
    float roughness_{};
    float metalness_{};
    float ior_{ 1.f };
    vec3 emissive_{};
    float translucent_{};

    uint32_t base_color_map_{ UINT32_MAX };
    uint32_t normal_map_{ UINT32_MAX };
    uint32_t roughness_map_{ UINT32_MAX };
    uint32_t metalness_map_{ UINT32_MAX };
    uint32_t emissive_map_{ UINT32_MAX };
};
