// The RobotStudio end of robot/station_feed.py: the snapshot frame on BoardOverlay, and its cloud.
//
// robot_trash_picker.py sends one request per snapshot -- on 'n' with the camera, at --frame with a
// rosbag -- naming two files built from the same frame:
//
//     floor   the annotated frame laid on the floor, 1 px = 1 mm, turned so +X runs right and +Y up
//     cloud   "TPC1", uint32 N, float32 N x 3 xyz in metres in the base frame, uint8 N x 3 RGB
//
// The picture goes on the part BoardOverlay and the cloud into the point cloud LiveCloud, in one
// call on the UI thread, so the station never shows the texture of one frame with the cloud of
// another. The station frame is the robot's base frame (the robot sits at its origin).
//
// Alignment: the texture coordinates of BoardOverlay are set here from its own vertices' world
// positions and the floor rectangle in the request -- the same formula as station_feed.floor_uv --
// rather than relying on a mapping applied by hand. A cloud point on the floor then sits on the
// texel that took its pixel's colour.
//
// The listener thread blocks in Accept and in ReadLine: nothing polls. One request at a time.
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Buffers.Binary;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using ABB.Robotics.Math;
using ABB.Robotics.RobotStudio;
using ABB.Robotics.RobotStudio.Stations;
using ABB.Robotics.RobotStudio.Stations.Forms;

namespace TrashPickerStation
{
    public static class StationAddin
    {
        // Mirrored in robot/station_feed.py; robot/tests/test_station_feed.py diffs the two.
        public const string StationHost = "127.0.0.1";
        public const int StationPort = 5003;
        const string CloudMagic = "TPC1";
        const string CloudName = "LiveCloud";
        const float CloudPointSize = 3.0f;
        const int ReadTimeoutMs = 10000;
        const string Category = "TrashPicker";

        static SynchronizationContext ui;
        static Thread listenerThread;
        // The station whose BoardOverlay material was last logged, so that is done once per station.
        static Station loggedStation;
        // What robot_trash_picker.py's --hide-cloud / --hide-rgb asked for, sent as it starts and
        // with every frame. A hidden one is still updated, so showing it again shows the last frame.
        static bool cloudVisible = true, floorVisible = true;

        public static void AddinMain()
        {
            ui = SynchronizationContext.Current;
            if (ui == null && System.Windows.Application.Current != null)
                ui = new System.Windows.Threading.DispatcherSynchronizationContext(
                    System.Windows.Application.Current.Dispatcher);
            if (ui == null)
            {
                Log("TrashPicker station: no UI thread to work on; the add-in is not running.",
                    LogMessageSeverity.Error);
                return;
            }
            listenerThread = new Thread(Listen) { IsBackground = true, Name = "TrashPicker station" };
            listenerThread.Start();
        }

        static void Listen()
        {
            TcpListener listener;
            try
            {
                listener = new TcpListener(IPAddress.Parse(StationHost), StationPort);
                listener.Start();
            }
            catch (SocketException failure)
            {
                Post($"TrashPicker station: cannot listen on {StationHost}:{StationPort}: " +
                     $"{failure.Message}. Is another RobotStudio already running it?",
                     LogMessageSeverity.Error);
                return;
            }
            Post($"TrashPicker station: listening on {StationHost}:{StationPort} for " +
                 "robot_trash_picker.py's snapshot (BoardOverlay and LiveCloud).",
                 LogMessageSeverity.Information);
            while (true)
            {
                TcpClient client;
                try { client = listener.AcceptTcpClient(); }
                catch (SocketException failure)
                {
                    // Not expected on a loopback listener; a pause keeps a persistent fault from
                    // spinning a core.
                    Post($"TrashPicker station: accept failed: {failure.Message}", LogMessageSeverity.Warning);
                    Thread.Sleep(1000);
                    continue;
                }
                using (client)
                {
                    try { Serve(client); }
                    catch (Exception failure)
                    {
                        // A client that went away mid-request ('h' in Python): nothing to answer.
                        Post($"TrashPicker station: request dropped: {failure.Message}",
                             LogMessageSeverity.Warning);
                    }
                }
            }
        }

        static void Serve(TcpClient client)
        {
            client.ReceiveTimeout = ReadTimeoutMs;
            var stream = client.GetStream();
            var reader = new StreamReader(stream, new UTF8Encoding(false));
            string line = reader.ReadLine();
            if (line == null)
                return;
            Dictionary<string, object> reply;
            try
            {
                using var request = JsonDocument.Parse(line);
                Dictionary<string, object> answer = null;
                Exception error = null;
                // RobotStudio's API is for the UI thread only.
                ui.Send(_ =>
                {
                    try { answer = Dispatch(request.RootElement); }
                    catch (Exception failure) { error = failure; }
                }, null);
                if (error != null)
                    ExceptionDispatchInfo.Capture(error).Throw();
                reply = answer;
            }
            catch (Exception failure)
            {
                Post($"TrashPicker station: {failure.Message}", LogMessageSeverity.Error);
                reply = new Dictionary<string, object> { ["ok"] = false, ["error"] = failure.Message };
            }
            byte[] bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(reply) + "\n");
            stream.Write(bytes, 0, bytes.Length);
        }

        // --- on the UI thread ----------------------------------------------------------------
        static Dictionary<string, object> Dispatch(JsonElement request)
        {
            string command = request.GetProperty("cmd").GetString();
            return command switch
            {
                "show" => Show(request),
                "screenshot" => Screenshot(request),
                "visibility" => Visibility(request),
                _ => throw new InvalidOperationException($"unknown command '{command}'"),
            };
        }

        static Dictionary<string, object> Show(JsonElement request)
        {
            Station station = Station.ActiveStation
                ?? throw new InvalidOperationException("no station is open in RobotStudio");
            string partName = request.GetProperty("part").GetString();
            Part part = FindPart(station, partName)
                ?? throw new InvalidOperationException($"the station has no part called {partName}");
            double[] floorX = Pair(request.GetProperty("floor_x_mm"));
            double[] floorY = Pair(request.GetProperty("floor_y_mm"));
            int frame = request.GetProperty("frame").GetInt32();

            // Both files are read before anything in the station changes: a bad file leaves the
            // last frame on screen, texture and cloud together.
            byte[] png = File.ReadAllBytes(request.GetProperty("floor").GetString());
            (Vector3[] points, Color[] colours) = ReadCloud(request.GetProperty("cloud").GetString());

            if (!ReferenceEquals(loggedStation, station))
            {
                LogMaterial(part);
                loggedStation = station;
            }
            // Every time, not once: the part may have been moved or replaced since, and it is one
            // face. Moving it then still keeps the picture on the floor rectangle.
            int faces = MapTexture(part, floorX, floorY);
            part.SetMaterial(new Material(new Texture(new MemoryStream(png))));

            PointCloud cloud = FindCloud(station);
            bool created = cloud == null;
            if (created)
                cloud = new PointCloud { Name = CloudName };
            ReadVisibility(request);
            part.Visible = floorVisible;
            cloud.Points = points;
            cloud.Colors = colours;
            cloud.PointSize = CloudPointSize;
            cloud.Visible = cloudVisible;
            if (created)
                station.PointClouds.Add(cloud);
            string view = null;
            if (request.TryGetProperty("look_down", out JsonElement look) && look.GetBoolean())
                view = LookDown(floorX, floorY);
            Station.UpdateGraphics(true);
            long releaseMs = ReleaseOldTextures();

            Log($"TrashPicker station: frame {frame} on {partName}, {points.Length} points in " +
                $"{CloudName}", LogMessageSeverity.Information);
            return new Dictionary<string, object>
            {
                ["ok"] = true, ["frame"] = frame, ["points"] = points.Length, ["mapped_faces"] = faces,
                ["view"] = view, ["undo_steps"] = Project.UndoContext.NumUndoSteps,
                ["release_ms"] = releaseMs,
            };
        }

        // Show or hide BoardOverlay and LiveCloud without a new frame: robot_trash_picker.py sends
        // this as it starts, with both stated, so a run undoes what an earlier one hid.
        static Dictionary<string, object> Visibility(JsonElement request)
        {
            Station station = Station.ActiveStation
                ?? throw new InvalidOperationException("no station is open in RobotStudio");
            string partName = request.GetProperty("part").GetString();
            Part part = FindPart(station, partName)
                ?? throw new InvalidOperationException($"the station has no part called {partName}");
            ReadVisibility(request);
            part.Visible = floorVisible;
            PointCloud cloud = FindCloud(station);
            if (cloud != null)
                cloud.Visible = cloudVisible;
            Station.UpdateGraphics(true);
            Log($"TrashPicker station: {partName} {(floorVisible ? "shown" : "hidden")}, " +
                $"{CloudName} {(cloudVisible ? "shown" : "hidden")}", LogMessageSeverity.Information);
            return new Dictionary<string, object>
            {
                ["ok"] = true, ["floor_visible"] = floorVisible, ["cloud_visible"] = cloudVisible,
                ["cloud_exists"] = cloud != null,
            };
        }

        // The flags a request carries; one it leaves out keeps what was asked for last.
        static void ReadVisibility(JsonElement request)
        {
            if (request.TryGetProperty("cloud_visible", out JsonElement cloud))
                cloudVisible = cloud.GetBoolean();
            if (request.TryGetProperty("floor_visible", out JsonElement floor))
                floorVisible = floor.GetBoolean();
        }

        // The 3D view as it is now, saved as a PNG: what the operator sees, for checking the
        // alignment without anyone at the screen. With look_down, after turning the view to the
        // floor rectangle in the request.
        static Dictionary<string, object> Screenshot(JsonElement request)
        {
            GraphicControl view = GraphicControl.ActiveGraphicControl
                ?? throw new InvalidOperationException("no active 3D view to take a screenshot of");
            string path = request.GetProperty("path").GetString();
            if (!Path.IsPathRooted(path) || !path.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"a screenshot goes to an absolute .png path, not {path}");
            string looked = null;
            if (request.TryGetProperty("look_down", out JsonElement look) && look.GetBoolean())
                looked = LookDown(Pair(request.GetProperty("floor_x_mm")),
                                  Pair(request.GetProperty("floor_y_mm")));
            Station.UpdateGraphics(true);
            int width = request.TryGetProperty("width", out JsonElement w) ? w.GetInt32() : 1600;
            int height = request.TryGetProperty("height", out JsonElement h) ? h.GetInt32() : 1000;
            using (Bitmap image = view.ScreenShot(width, height))
                image.Save(path, System.Drawing.Imaging.ImageFormat.Png);
            return new Dictionary<string, object> { ["ok"] = true, ["path"] = path, ["view"] = looked };
        }

        // Straight down on the floor rectangle, +X to the right and +Y up, as the picture is laid
        // out: for the test pattern (python -m robot.station_feed --test-pattern), not for a run,
        // which leaves the operator's view alone.
        static string LookDown(double[] floorX, double[] floorY)
        {
            GraphicControl view = GraphicControl.ActiveGraphicControl;
            if (view == null)
                return "no active 3D view";
            double cx = (floorX[0] + floorX[1]) / 2000.0, cy = (floorY[0] + floorY[1]) / 2000.0;
            try
            {
                // UpDirection first: the default +Z would be parallel to a straight-down look.
                var camera = new Camera
                {
                    UpDirection = new Vector3(0.0, 1.0, 0.0),
                    LookFrom = new Vector3(cx, cy, 2.5), LookAt = new Vector3(cx, cy, 0.0),
                    Projection = GraphicProjection.Perspective, FieldOfView = view.FieldOfView,
                };
                view.Camera = camera;
                view.SyncCamera(camera, false, 0f);
                view.Update();
                Vector3 at = view.Camera.LookFrom;
                return string.Format(CultureInfo.InvariantCulture, "looking down from ({0:F2}, {1:F2}, {2:F2})",
                                     at.x, at.y, at.z);
            }
            catch (Exception failure)
            {
                Log($"TrashPicker station: could not turn the view to the floor: {failure.Message}",
                    LogMessageSeverity.Warning);
                return failure.Message;
            }
        }

        // A Texture has no Dispose: its native image (1420 x 770 x 4 bytes, about 4 MB) goes only
        // when its finalizer runs. A replaced texture is a few bytes of managed memory, so nothing
        // makes the collector run, and RobotStudio grew by one picture per frame shown. One
        // collection per frame shown -- per 'n', not per camera frame -- frees the one just
        // replaced. Measured, and returned in the reply as release_ms.
        static long ReleaseOldTextures()
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            return clock.ElapsedMilliseconds;
        }

        static Part FindPart(Station station, string name)
        {
            foreach (GraphicComponent component in station.FindGraphicComponentsByType(typeof(Part)))
                if (component is Part part && part.Name == name)
                    return part;
            return null;
        }

        static PointCloud FindCloud(Station station)
        {
            foreach (PointCloud cloud in station.PointClouds)
                if (cloud.Name == CloudName)
                    return cloud;
            return null;
        }

        // Every vertex of every face gets u along X and v down from +Y max, from its world position
        // in millimetres: the inverse of how station_feed.py lays the picture out.
        static int MapTexture(Part part, double[] floorX, double[] floorY)
        {
            double xMin = floorX[0], width = floorX[1] - floorX[0];
            double yMax = floorY[1], height = floorY[1] - floorY[0];
            Matrix4 toWorld = part.Transform.GlobalMatrix;
            int faces = 0;
            foreach (object level in part.Mesh)
            {
                if (level is not MeshPart meshPart)
                    continue;
                foreach (MeshFace face in meshPart.GetAllFaces())
                {
                    var uvs = new List<Vector2>(face.Vertices.Count);
                    foreach (Vector3 vertex in face.Vertices)
                    {
                        Vector3 world = toWorld.MultiplyPoint(vertex);
                        double x = world.x * 1000.0, y = world.y * 1000.0;
                        uvs.Add(new Vector2((x - xMin) / width, (yMax - y) / height));
                    }
                    face.TextureCoordinates.Clear();
                    face.TextureCoordinates.AddRange(uvs);
                    faces++;
                }
            }
            if (faces == 0)
                throw new InvalidOperationException($"{part.Name} has no mesh faces to put the picture on");
            part.Mesh.Rebuild();
            return faces;
        }

        static (Vector3[], Color[]) ReadCloud(string path)
        {
            byte[] data = File.ReadAllBytes(path);
            if (data.Length < 8 || Encoding.ASCII.GetString(data, 0, 4) != CloudMagic)
                throw new InvalidDataException($"{path} is not a station cloud");
            long count = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(4));
            if (data.Length != 8 + count * 15)
                throw new InvalidDataException($"{path} holds {data.Length} bytes, not {count} points");
            int rgb = 8 + (int)count * 12;
            var points = new Vector3[count];
            var colours = new Color[count];
            for (int i = 0; i < count; i++)
            {
                int at = 8 + i * 12;
                points[i] = new Vector3(BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(at)),
                                        BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(at + 4)),
                                        BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(at + 8)));
                int c = rgb + i * 3;
                colours[i] = Color.FromArgb(data[c], data[c + 1], data[c + 2]);
            }
            return (points, colours);
        }

        static double[] Pair(JsonElement element)
        {
            if (element.GetArrayLength() != 2)
                throw new InvalidDataException("a floor range needs two numbers");
            return new[] { element[0].GetDouble(), element[1].GetDouble() };
        }

        static void LogMaterial(Part part)
        {
            try
            {
                Material material = part.GetMaterial();
                string texture = material?.BaseTexture == null ? "none" : material.BaseTexture.Size.ToString();
                Log(string.Format(CultureInfo.InvariantCulture,
                    "TrashPicker station: {0} had texture {1}, TextureSize {2}, TextureMatrix identity {3}; " +
                    "its texture coordinates are now set from its vertices.",
                    part.Name, texture, material?.TextureSize, material?.TextureMatrix.IsIdentity()),
                    LogMessageSeverity.Information);
            }
            catch (Exception failure)
            {
                Log($"TrashPicker station: could not read {part.Name}'s material: {failure.Message}",
                    LogMessageSeverity.Warning);
            }
        }

        static void Log(string text, LogMessageSeverity severity) =>
            Logger.AddMessage(new LogMessage(text, Category, severity));

        // From the listener thread: the Output window belongs to the UI thread too.
        static void Post(string text, LogMessageSeverity severity) =>
            ui.Post(_ => Log(text, severity), null);
    }
}
