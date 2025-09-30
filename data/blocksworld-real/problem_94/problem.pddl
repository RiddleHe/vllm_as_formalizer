(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        purple - block
        red - block
        yellow - block
        green - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on purple blue)
        (clear purple)
        (ontable red)
        (on yellow red)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        ; blue -> purple
        ; red -> yellow
        ; green
        ; orange
    )
    (:goal (and (on orange green)
        (on yellow orange)
        (on blue yellow)
        (on red blue)
        (on purple red)))
)
    