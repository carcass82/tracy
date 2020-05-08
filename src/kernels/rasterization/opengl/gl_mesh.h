/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "gl_material.h"

struct GLMesh
{
    GLMesh(const Mesh& mesh)
        : material(*mesh.GetMaterial())
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vb;
        glGenBuffers(1, &vb);
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        glBufferData(GL_ARRAY_BUFFER, mesh.GetVertexCount() * sizeof(Vertex), &mesh.GetVertices()[0], GL_STATIC_DRAW);
        vertexcount = mesh.GetVertexCount();

        GLuint ib;
        glGenBuffers(1, &ib);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.GetIndexCount() * sizeof(Index), &mesh.GetIndices()[0], GL_STATIC_DRAW);
        indexcount = mesh.GetIndexCount();

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

        glDeleteBuffers(1, &vb);
        glDeleteBuffers(1, &ib);
    }

    ~GLMesh()
    {
        glDeleteVertexArrays(1, &vao);
    }

    GLMesh(const GLMesh&) = delete;
    GLMesh& operator=(const GLMesh&) = delete;

    GLMesh(GLMesh&& other)
        : vao(std::exchange(other.vao, 0))
        , vertexcount(std::exchange(other.vertexcount, 0))
        , indexcount(std::exchange(other.indexcount, 0))
        , material(other.material)
    {}
    
    GLMesh& operator=(GLMesh&& other)
    {
        vao = std::exchange(other.vao, 0);
        vertexcount = std::exchange(other.vertexcount, 0);
        indexcount = std::exchange(other.indexcount, 0);
        material = other.material;

        return *this;
    }

    GLuint vao;
    uint32_t vertexcount;
    uint32_t indexcount;
    GLMaterial material;
};
