(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        green - block
        blue - block
        orange - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on purple red)
        (clear purple)
        (ontable green)
        (clear green)
        (ontable blue)
        (on orange blue)
        (on yellow orange)
        (clear yellow)
        ; red -> purple
        ; green
        ; blue -> orange -> yellow
    )
    (:goal (and (on blue red)
        (on yellow blue)
        (on orange yellow)
        (on purple orange)
        (on green purple)))
)
    