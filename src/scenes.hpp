#pragma once

hitable* random_scene()
{
    const int n = 500;
    hitable** list = new hitable*[n + 1];

    texture* terrain_texture = new checker_texture(new constant_texture(glm::vec3(0.2, 0.3, 0.1)), new constant_texture(glm::vec3(0.9, 0.9, 0.9)));
    list[0] = new sphere(glm::vec3(0, -1000, 0), 1000, new lambertian(terrain_texture));

    int i = 1;
    for (int a = -10; a < 10; ++a) {
        for (int b = -10; b < 10; ++b) {
            float choose_mat = float(drand48());
            glm::vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
            if ((center - glm::vec3(4, 0.2, 0)).length() > 0.9) {
                if (choose_mat < 0.8) {
                    //list[i++] = new moving_sphere(center, center + vec3(0, 0.5 * drand48(), 0.0), 0.0, 1.0, 0.2, new lambertian(new constant_texture(vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48())));
                    list[i++] = new sphere(center, 0.2f, new lambertian(new constant_texture(glm::vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48()))));
                } else if (choose_mat < 0.95) {
                    list[i++] = new sphere(center, 0.2f, new metal(glm::vec3(0.5f * (1.0f + float(drand48())), 0.5f * (1.0f + float(drand48())), 0.5f * (1.0f + float(drand48()))), 0.5f * float(drand48())));
                } else {
                    list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                }

            }
        }
    }

    // area light
    //list[i++] = new xy_rect(-2, 6, 0, 3, -3, new diffuse_light(new constant_texture(vec3(4,4,4))));
    list[i++] = new xz_rect(-20, 20, -20, 20, 10, new diffuse_light(new constant_texture(glm::vec3(2,2,2))));

    // lambertian
    list[i++] = new sphere(glm::vec3(-2, 1, 0), 1.0, new lambertian(new constant_texture(glm::vec3(0.4, 0.2, 0.1))));

    // dielectric
    list[i++] = new sphere(glm::vec3(0, 1, 0), 1.0, new dielectric(1.5));

    // metal
    list[i++] = new sphere(glm::vec3(2, 1, 0), 1.0, new metal(glm::vec3(0.7, 0.6, 0.5), 0.0f));

    // lambertian noise ("marble like")
    list[i++] = new sphere(glm::vec3(4, 1, 0), 1.0, new lambertian(new noise_texture(5.0f)));

    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    list[i++] = new sphere(glm::vec3(6, 1, 0), 1.0, new lambertian(new image_texture(tex_data, nx, ny)));

    return new hitable_list(list, i);
}

hitable* cornellbox_scene()
{
    const int n = 50;
    hitable** list = new hitable*[n + 1];

    material* red = new lambertian(new constant_texture(glm::vec3(0.65, 0.05, 0.05)));
    material* white = new lambertian(new constant_texture(glm::vec3(0.73, 0.73, 0.73)));
    material* green = new lambertian(new constant_texture(glm::vec3(0.12, 0.45, 0.15)));
    material* light = new diffuse_light(new constant_texture(glm::vec3(15, 15, 15)));
    //material* glass = new dielectric(1.5);

    int i = 0;
    list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
    list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
    list[i++] = new xz_rect(203, 353, 237, 322, 548, light);
    list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
    list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
    list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

    list[i++] = new translate(new rotate_y(new box(glm::vec3(0, 0, 0), glm::vec3(165, 165, 165), white), -18), glm::vec3(130, 0, 65));
    list[i++] = new translate(new rotate_y(new box(glm::vec3(0, 0, 0), glm::vec3(165, 330, 165), white), 15), glm::vec3(265, 0, 295));
    list[i++] = new sphere(glm::vec3(190, 50, 100), 50.0, new dielectric(1.5));

    //list[i++] = new flip_normals(new xz_rect(203, 353, 237, 322, 1, light));

    return new hitable_list(list, i);
}

hitable* final()
{
    int nb = 20;
    hitable** list = new hitable*[30];
    hitable** boxlist = new hitable*[10000];
    hitable** boxlist2 = new hitable*[10000];

    material* white = new lambertian(new constant_texture(glm::vec3(0.73, 0.73, 0.73)));
    material* ground = new lambertian(new constant_texture(glm::vec3(0.48, 0.83, 0.53)));
    material* light = new diffuse_light(new constant_texture(glm::vec3(15, 15, 15)));

    int b = 0;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            float w = 100;
            float x0 = -1000 + i * w;
            float z0 = -1000 + j * w;
            float y0 = 0;
            float x1 = x0 + w;
            float y1 = 100 * (float(drand48()) + 0.01f);
            float z1 = z0 + w;

            boxlist[b++] = new box(glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1), ground);
        }
    }

    //material* _lambertian = new lambertian(new constant_texture(glm::vec3(0.7, 0.3, 0.1)));
    material* _dielectric = new dielectric(1.5);
    material* _metal = new metal(glm::vec3(0.8, 0.8, 0.9), 10.0);

    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    material* _textured = new lambertian(new image_texture(tex_data, nx, ny));

    material* _noise = new lambertian(new noise_texture(0.1));


    int l = 0;
    list[l++] = new bvh_node(boxlist, b, 0, 1);
    list[l++] = new xz_rect(123, 423, 147, 412, 554, light);
    //list[l++] = new moving_sphere(vec3(400,400,200), vec3(430,400,200), 0, 1, 50, _lambertian);
    list[l++] = new sphere(glm::vec3(260, 150, 45), 50, _dielectric);
    list[l++] = new sphere(glm::vec3(0, 150, 145), 50, _metal);

    hitable* boundary = new sphere(glm::vec3(360, 150, 145), 70, _dielectric);
    list[l++] = boundary;
    //list[l++] = new constant_medium(boundary, 0.2, new constant_texture(vec3(0.2, 0.4, 0.9)));

    boundary = new sphere(glm::vec3(0, 0, 0), 5000, _dielectric);
    //list[l++] = new constant_medium(boundary, 0.0001, new constant_texture(vec3(1.0, 1.0, 1.0)));

    list[l++] = new sphere(glm::vec3(400, 200, 400), 100, _textured);
    list[l++] = new sphere(glm::vec3(220, 280, 300), 80, _noise);

    int ns = 1000;
    for (int i = 0; i < ns; ++i) {
        boxlist2[i] = new sphere(glm::vec3(165 * drand48(), 165 * drand48(), 165 * drand48()), 10, white);
    }

    list[l++] = new translate(new rotate_y(new bvh_node(boxlist2, ns, 0.0, 1.0), 15), glm::vec3(-100, 270, 395));

    return new hitable_list(list, l);
}

hitable* test_scene()
{
    hitable** list = new hitable*[30];


    //texture* ground = new checker_texture(new constant_texture(glm::vec3(0.2, 0.2, 0.4)), new constant_texture(glm::vec3(0.8, 0.8, 1.0)));
    //material* _lambertian = new lambertian(ground);
    //material* red = new lambertian(new constant_texture(glm::vec3(0.65, 0.05, 0.05)));
    material* _light = new diffuse_light(new constant_texture(glm::vec3(15, 15, 10)));
    
    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    
    int i = 0;

    list[i++] = new sphere(glm::vec3(1, 0.6, 0), 0.3, _light);
    //list[i++] = new sphere(glm::vec3(0, -1, -1), 1, _lambertian);
    //list[i++] = new sphere(glm::vec3(0, -1, -1), 1.0, new lambertian(new noise_texture(5.0f)));
    //list[i++] = new xz_rect(-50, 50, -50, 50, 0, red);
    //list[i++] = new yz_rect(0, 50, 0, 50, 0, red);
    list[i++] = new sphere(glm::vec3(0, -1, -1), 1.0f, new lambertian(new image_texture(tex_data, nx, ny)));

    return new hitable_list(list, i);
}


enum eScene { eRANDOM, eCORNELLBOX, eFINAL, eTEST, eFROMFILE, eNUM_SCENES };

hitable* load_scene(eScene scene, camera& cam, float ratio)
{
    switch (scene) {
    case eRANDOM:
        std::cerr << "tracing random scene...\n";
        cam.setup(glm::vec3(10.0f, 1.5f, 4.0f), glm::vec3(2.0f, 0.5f, -2.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, ratio, 2.0f, 5.0f);
        return random_scene();
        
    case eCORNELLBOX:
        std::cerr << "tracing cornell scene...\n";
        cam.setup(glm::vec3(278, 278, -800), glm::vec3(278, 278, 0), glm::vec3(0.0f, 1.0f, 0.0f), 40.0f, ratio, 0.0f, 10.0f);
        return cornellbox_scene();
        
    case eFINAL:
        std::cerr << "tracing final scene...\n";
        cam.setup(glm::vec3(278, 278, -800), glm::vec3(278, 278, 0), glm::vec3(0.0f, 1.0f, 0.0f), 40.0f, ratio, 0.0f, 10.0f);
        return final();

    case eTEST:
        std::cerr << "tracing test scene...\n";
        cam.setup(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 45.0f, ratio, 0.0f, 10.0f);
        return test_scene();
        
    default:
        std::cerr << "tracing NULL, i'm going to crash...\n";
        return nullptr;
    };
}