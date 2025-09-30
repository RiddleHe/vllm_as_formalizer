(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        blue - block
        orange - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        (ontable blue)
        (on orange blue)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        ; red
        ; purple
        ; blue -> orange
        ; yellow
        ; green
    )
    (:goal (and (on blue green)
        (on red blue)
        (on orange red)
        (on yellow orange)
        (on purple yellow)))
)
    