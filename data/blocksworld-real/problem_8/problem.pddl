(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        purple - block
        orange - block
        blue - block
        red - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable purple)
        (clear purple)
        (ontable orange)
        (on blue orange)
        (clear blue)
        (ontable red)
        (clear red)
        (ontable yellow)
        (clear yellow)
        ; green
        ; purple
        ; orange -> blue
        ; red
        ; yellow
    )
    (:goal (and (on blue purple)
        (on yellow blue)
        (on orange green)
        (on red orange)))
)
    