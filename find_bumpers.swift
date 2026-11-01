// Finds team numbers on robot bumpers in the wide field shot with Apple Vision.
// Usage: swift find_bumpers.swift field/ 868,2910,2046,4065,4414,1323 2  -> CSV: video_s,team,x,y,confidence
import Foundation
import Vision
import AppKit

let dir = CommandLine.arguments[1]
let teams = Set(CommandLine.arguments[2].split(separator: ",").map(String.init))
let fps = Double(CommandLine.arguments[3]) ?? 2
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
print("video_s,team,x,y,confidence")
for (i, f) in files.enumerated() {
    guard let img = NSImage(contentsOf: URL(fileURLWithPath: dir).appendingPathComponent(f))?
        .cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.minimumTextHeight = 0.02
    try VNImageRequestHandler(cgImage: img).perform([req])
    for obs in req.results ?? [] {
        for cand in obs.topCandidates(3) {
            let digits = cand.string.filter(\.isNumber)
            if teams.contains(digits) {
                let b = obs.boundingBox
                print(String(format: "%.1f,%@,%.0f,%.0f,%.2f", 3 + Double(i) / fps, digits,
                             b.midX * 1920, 160 + (1 - b.midY) * 540, cand.confidence))
                break
            }
        }
    }
}
