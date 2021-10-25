/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "GL/glew.h"
#include "gl_material.h"
#include "mesh.h"

class GLMesh
{
public:

    GLMesh()
    {
    }

    GLMesh(const Mesh& mesh, const Material& material)
        : vao_{}
        , indexcount_{ mesh.GetIndexCount() }
        , material_{ material }
    {
        glGenVertexArrays(1, &vao_);
        glBindVertexArray(vao_);

        GLuint vb;
        glGenBuffers(1, &vb);
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        glBufferData(GL_ARRAY_BUFFER, mesh.GetVertexCount() * sizeof(Vertex), mesh.GetVertices(), GL_STATIC_DRAW);

        GLuint ib;
        glGenBuffers(1, &ib);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexcount_ * sizeof(Index), mesh.GetIndices(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, uv0));
        glEnableVertexAttribArray(2);

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, tangent));
        glEnableVertexAttribArray(3);

        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, bitangent));
        glEnableVertexAttribArray(4);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &vb);
        glDeleteBuffers(1, &ib);
    }

    ~GLMesh()
    {
        glDeleteVertexArrays(1, &vao_);
    }

    GLMesh(const GLMesh&) = delete;
    GLMesh& operator=(const GLMesh&) = delete;

    GLMesh(GLMesh&& other) noexcept
        : vao_{ std::exchange(other.vao_, 0) }
        , indexcount_{ std::exchange(other.indexcount_, 0) }
        , material_{ other.material_ }
    {
    }

    GLMesh& operator=(GLMesh&& other) noexcept
    {
		if (this != &other)
		{
			vao_ = std::exchange(other.vao_, 0);
			indexcount_ = std::exchange(other.indexcount_, 0);
			material_ = other.material_;
		}
		
		return *this;
    }

    const GLMaterial& GetMaterial() const { return material_; }

    void Draw() const
    {
        glBindVertexArray(vao_);
        glDrawElements(GL_TRIANGLES, indexcount_, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

private:
    GLuint vao_{};
    uint32_t indexcount_{};
    GLMaterial material_{};
};
